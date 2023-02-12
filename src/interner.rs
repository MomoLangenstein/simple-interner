use {
    crate::Interned,
    core::{
        borrow::Borrow,
        cell::UnsafeCell,
        cmp::Ordering,
        fmt,
        hash::{BuildHasher, Hash, Hasher},
        marker::PhantomData,
        ops::Deref,
        ptr::NonNull,
    },
};

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(feature = "std")]
use std::collections::hash_map::RandomState;

#[cfg(feature = "raw")]
use hashbrown::hash_map::RawEntryMut;
use hashbrown::hash_map::{Entry, HashMap};

use crate::LeakedInterner;

/// A wrapper around box that does not provide &mut access to the pointee and
/// uses raw-pointer borrowing rules to avoid invalidating extant references.
///
/// The resolved reference is guaranteed valid until the PinBox is dropped.
struct PinBox<T: ?Sized> {
    ptr: NonNull<T>,
    _marker: PhantomData<Box<T>>,
}

impl<T: ?Sized> PinBox<T> {
    fn new(x: Box<T>) -> Self {
        Self {
            ptr: NonNull::new(Box::into_raw(x)).unwrap(),
            _marker: PhantomData,
        }
    }

    #[allow(unsafe_code)]
    unsafe fn as_ref<'a>(&self) -> &'a T {
        self.ptr.as_ref()
    }
}

impl<T: ?Sized> Drop for PinBox<T> {
    fn drop(&mut self) {
        #[allow(unsafe_code)] // SAFETY: PinBox acts like Box.
        unsafe {
            Box::from_raw(self.ptr.as_ptr())
        };
    }
}

impl<T: ?Sized> Deref for PinBox<T> {
    type Target = T;
    #[allow(unsafe_code)] // SAFETY: PinBox acts like Box.
    fn deref(&self) -> &T {
        unsafe { self.as_ref() }
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for PinBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl<T: ?Sized + Eq> Eq for PinBox<T> {}
impl<T: ?Sized + PartialEq> PartialEq for PinBox<T> {
    fn eq(&self, other: &Self) -> bool {
        (**self).eq(&**other)
    }
}
impl<T: ?Sized + PartialEq> PartialEq<T> for PinBox<T> {
    fn eq(&self, other: &T) -> bool {
        (**self).eq(other)
    }
}

impl<T: ?Sized + Ord> Ord for PinBox<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        (**self).cmp(&**other)
    }
}
impl<T: ?Sized + PartialOrd> PartialOrd for PinBox<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }
}
impl<T: ?Sized + PartialOrd> PartialOrd<T> for PinBox<T> {
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        (**self).partial_cmp(other)
    }
}

impl<T: ?Sized + Hash> Hash for PinBox<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T: ?Sized> Borrow<T> for PinBox<T> {
    fn borrow(&self) -> &T {
        self
    }
}

#[allow(unsafe_code)] // SAFETY: PinBox acts like Box.
unsafe impl<T: ?Sized> Send for PinBox<T> where Box<T>: Send {}

#[allow(unsafe_code)] // SAFETY: PinBox acts like Box.
unsafe impl<T: ?Sized> Sync for PinBox<T> where Box<T>: Sync {}

pub trait InternerUtils {
    const CONST_INIT: Self;
}

impl InternerUtils for () {
    const CONST_INIT: Self = ();
}

pub(crate) type InternerState = ();

pub trait InternerLock<T: InternerUtils> {
    const CONST_INIT: Self;
    fn new(value: T) -> Self;
    fn try_with_ref<Q>(&self, inner: impl FnOnce(&T) -> Q) -> Option<Q>;
    fn with_ref<Q>(&self, inner: impl FnOnce(&T) -> Q) -> Q;
    fn with_ref_mut<Q>(&self, inner: impl FnOnce(&mut T) -> Q) -> Q;
    fn with_mut_ref_mut<Q>(&mut self, inner: impl FnOnce(&mut T) -> Q) -> Q;
}

impl<T: InternerUtils> InternerLock<T> for UnsafeCell<T> {
    const CONST_INIT: Self = Self::new(T::CONST_INIT);

    fn new(value: T) -> Self {
        Self::new(value)
    }

    fn try_with_ref<Q>(&self, inner: impl FnOnce(&T) -> Q) -> Option<Q> {
        #[allow(unsafe_code)]
        Some(inner(unsafe { &*self.get() }))
    }

    fn with_ref<Q>(&self, inner: impl FnOnce(&T) -> Q) -> Q {
        #[allow(unsafe_code)]
        inner(unsafe { &*self.get() })
    }

    fn with_ref_mut<Q>(&self, inner: impl FnOnce(&mut T) -> Q) -> Q {
        #[allow(unsafe_code)]
        inner(unsafe { &mut *self.get() })
    }

    fn with_mut_ref_mut<Q>(&mut self, inner: impl FnOnce(&mut T) -> Q) -> Q {
        inner(self.get_mut())
    }
}

#[cfg(feature = "std")]
impl<T: InternerUtils> InternerLock<T> for std::sync::RwLock<T> {
    const CONST_INIT: Self = Self::new(T::CONST_INIT);

    fn new(value: T) -> Self {
        Self::new(value)
    }

    fn try_with_ref<Q>(&self, inner: impl FnOnce(&T) -> Q) -> Option<Q> {
        let read = match self.try_read() {
            Ok(read) => read,
            Err(std::sync::TryLockError::WouldBlock) => return None,
            r @ Err(std::sync::TryLockError::Poisoned(_)) => {
                r.expect("interner lock should not be poisoned")
            },
        };

        Some(inner(&*read))
    }

    fn with_ref<Q>(&self, inner: impl FnOnce(&T) -> Q) -> Q {
        let read = self.read().expect("interner lock should not be poisoned");

        inner(&*read)
    }

    fn with_ref_mut<Q>(&self, inner: impl FnOnce(&mut T) -> Q) -> Q {
        let mut write = self.write().expect("interner lock should not be poisoned");

        inner(&mut *write)
    }

    fn with_mut_ref_mut<Q>(&mut self, inner: impl FnOnce(&mut T) -> Q) -> Q {
        let r#mut = self
            .get_mut()
            .expect("interner lock should not be poisoned");

        inner(r#mut)
    }
}

#[cfg(feature = "std")]
impl<T: InternerUtils> InternerLock<T> for std::sync::Mutex<T> {
    const CONST_INIT: Self = Self::new(T::CONST_INIT);

    fn new(value: T) -> Self {
        Self::new(value)
    }

    fn try_with_ref<Q>(&self, inner: impl FnOnce(&T) -> Q) -> Option<Q> {
        let guard = match self.try_lock() {
            Ok(read) => read,
            Err(std::sync::TryLockError::WouldBlock) => return None,
            r @ Err(std::sync::TryLockError::Poisoned(_)) => {
                r.expect("interner lock should not be poisoned")
            },
        };

        Some(inner(&*guard))
    }

    fn with_ref<Q>(&self, inner: impl FnOnce(&T) -> Q) -> Q {
        let guard = self.lock().expect("interner lock should not be poisoned");

        inner(&*guard)
    }

    fn with_ref_mut<Q>(&self, inner: impl FnOnce(&mut T) -> Q) -> Q {
        let mut guard = self.lock().expect("interner lock should not be poisoned");

        inner(&mut *guard)
    }

    fn with_mut_ref_mut<Q>(&mut self, inner: impl FnOnce(&mut T) -> Q) -> Q {
        let r#mut = self
            .get_mut()
            .expect("interner lock should not be poisoned");

        inner(r#mut)
    }
}

impl<R: lock_api::RawRwLock, T: InternerUtils> InternerLock<T> for lock_api::RwLock<R, T> {
    const CONST_INIT: Self = Self::new(T::CONST_INIT);

    fn new(value: T) -> Self {
        Self::new(value)
    }

    fn try_with_ref<Q>(&self, inner: impl FnOnce(&T) -> Q) -> Option<Q> {
        let read = self.try_read()?;

        Some(inner(&*read))
    }

    fn with_ref<Q>(&self, inner: impl FnOnce(&T) -> Q) -> Q {
        let read = self.read();

        inner(&*read)
    }

    fn with_ref_mut<Q>(&self, inner: impl FnOnce(&mut T) -> Q) -> Q {
        let mut write = self.write();

        inner(&mut *write)
    }

    fn with_mut_ref_mut<Q>(&mut self, inner: impl FnOnce(&mut T) -> Q) -> Q {
        inner(self.get_mut())
    }
}

impl<R: lock_api::RawMutex, T: InternerUtils> InternerLock<T> for lock_api::Mutex<R, T> {
    const CONST_INIT: Self = Self::new(T::CONST_INIT);

    fn new(value: T) -> Self {
        Self::new(value)
    }

    fn try_with_ref<Q>(&self, inner: impl FnOnce(&T) -> Q) -> Option<Q> {
        let guard = self.try_lock()?;

        Some(inner(&*guard))
    }

    fn with_ref<Q>(&self, inner: impl FnOnce(&T) -> Q) -> Q {
        let guard = self.lock();

        inner(&*guard)
    }

    fn with_ref_mut<Q>(&self, inner: impl FnOnce(&mut T) -> Q) -> Q {
        let mut guard = self.lock();

        inner(&mut *guard)
    }

    fn with_mut_ref_mut<Q>(&mut self, inner: impl FnOnce(&mut T) -> Q) -> Q {
        inner(self.get_mut())
    }
}

#[cfg(feature = "std")]
/// An interner based on a `HashSet`. See the crate-level docs for more.
pub struct Interner<
    T: ?Sized,
    S = RandomState,
    L: InternerLock<InternerState> = std::sync::RwLock<InternerState>,
> {
    lock: L,
    arena: UnsafeCell<HashMap<PinBox<T>, (), S>>,
}

#[cfg(not(feature = "std"))]
/// An interner based on a `HashSet`. See the crate-level docs for more.
pub struct Interner<T: ?Sized, S, L: InternerLock<InternerState>> {
    lock: L,
    arena: UnsafeCell<HashMap<PinBox<T>, (), S>>,
}

#[allow(unsafe_code)]
unsafe impl<T: ?Sized + Send, S: Send, L: InternerLock<InternerState> + Send> Send
    for Interner<T, S, L>
{
}

#[allow(unsafe_code)]
unsafe impl<T: ?Sized + Sync, S: Sync, L: InternerLock<InternerState> + Sync> Sync
    for Interner<T, S, L>
{
}

impl<T: ?Sized + fmt::Debug, S, L: InternerLock<InternerState>> fmt::Debug for Interner<T, S, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct InternerFormatter<'a, T: ?Sized + fmt::Debug, S, L: InternerLock<InternerState>>(
            &'a Interner<T, S, L>,
        );

        impl<'a, T: ?Sized + fmt::Debug, S, L: InternerLock<InternerState>> fmt::Debug
            for InternerFormatter<'a, T, S, L>
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0
                    .lock
                    .try_with_ref(|_guard| {
                        #[allow(unsafe_code)] // SAFETY: access is guarded by the lock
                        let arena = unsafe { &*self.0.arena.get() };

                        f.debug_set().entries(arena.keys()).finish()
                    })
                    .unwrap_or_else(|| f.write_str("<locked>"))
            }
        }

        f.debug_struct("Interner")
            .field("arena", &InternerFormatter(self))
            .finish()
    }
}

impl<T: ?Sized, S: Default, L: InternerLock<InternerState> + Default> Default
    for Interner<T, S, L>
{
    fn default() -> Self {
        Interner {
            lock: L::default(),
            arena: UnsafeCell::new(HashMap::default()),
        }
    }
}

impl<T: Eq + Hash + ?Sized, S: BuildHasher, L: InternerLock<InternerState>> Interner<T, S, L> {
    /// Intern an item into the interner.
    ///
    /// Takes borrowed or heap-allocated items. If the item has not been
    /// previously interned, it will be `Into::into`ed a `Box` on the heap and
    /// cached. Notably, if you give this fn a `String` or `Vec`, the allocation
    /// will be shrunk to fit.
    ///
    /// Note that the interner may need to reallocate to make space for the new
    /// reference, just the same as a `HashSet` would. This cost is amortized to
    /// `O(1)` as it is in other standard library collections.
    ///
    /// If you have an owned item (especially if it has a cheap transformation
    /// to `Box`) and no longer need the ownership, pass it in directly.
    /// Otherwise, pass in a reference.
    ///
    /// See `get` for more about the interned symbol.
    pub fn intern_mut<B>(&mut self, t: B) -> Interned<'_, T>
    where
        B: Borrow<T> + Into<Box<T>>,
    {
        self.lock.with_mut_ref_mut(|_guard| {
            let arena = self.arena.get_mut();

            if let Some((t, _)) = arena.get_key_value(t.borrow()) {
                #[allow(unsafe_code)] // SAFETY: Interned ties the lifetime to the interner.
                return Interned(unsafe { t.as_ref() });
            }

            // If someone interned the item between the above check and us acquiring
            // the write lock, this heap allocation isn't necessary. However, this
            // is expected to be rare, so we don't bother with doing another lookup
            // before creating the box. Using the raw_entry API could avoid this,
            // but needs a different call than intern_raw to use the intrinsic
            // BuildHasher rather than an external one. It's not worth the effort.

            let entry = arena.entry(PinBox::new(t.into()));
            #[allow(unsafe_code)] // SAFETY: Interned ties the lifetime to the interner.
            match entry {
                Entry::Occupied(entry) => Interned(unsafe { entry.key().as_ref() }),
                Entry::Vacant(entry) => {
                    let interned = Interned(unsafe { entry.key().as_ref() });
                    entry.insert(());
                    interned
                },
            }
        })
    }

    /// Intern an item into the interner.
    ///
    /// Takes borrowed or heap-allocated items. If the item has not been
    /// previously interned, it will be `Into::into`ed a `Box` on the heap and
    /// cached. Notably, if you give this fn a `String` or `Vec`, the allocation
    /// will be shrunk to fit.
    ///
    /// Note that the interner may need to reallocate to make space for the new
    /// reference, just the same as a `HashSet` would. This cost is amortized to
    /// `O(1)` as it is in other standard library collections.
    ///
    /// If you have an owned item (especially if it has a cheap transformation
    /// to `Box`) and no longer need the ownership, pass it in directly.
    /// Otherwise, pass in a reference.
    ///
    /// See `get` for more about the interned symbol.
    pub fn intern<B>(&self, t: B) -> Interned<'_, T>
    where
        B: Borrow<T> + Into<Box<T>>,
    {
        if let Some(interned) = self.get(t.borrow()) {
            return interned;
        }

        self.lock.with_ref_mut(|_guard| {
            #[allow(unsafe_code)] // SAFETY: access is guarded by the lock
            let arena = unsafe { &mut *self.arena.get() };

            // If someone interned the item between the above check and us acquiring
            // the write lock, this heap allocation isn't necessary. However, this
            // is expected to be rare, so we don't bother with doing another lookup
            // before creating the box. Using the raw_entry API could avoid this,
            // but needs a different call than intern_raw to use the intrinsic
            // BuildHasher rather than an external one. It's not worth the effort.

            let entry = arena.entry(PinBox::new(t.into()));
            #[allow(unsafe_code)] // SAFETY: Interned ties the lifetime to the interner.
            match entry {
                Entry::Occupied(entry) => Interned(unsafe { entry.key().as_ref() }),
                Entry::Vacant(entry) => {
                    let interned = Interned(unsafe { entry.key().as_ref() });
                    entry.insert(());
                    interned
                },
            }
        })
    }

    /// Get an interned reference out of this interner.
    ///
    /// The returned reference is bound to the lifetime of the borrow used for
    /// this method. This guarantees that the returned reference will live no
    /// longer than this interner does.
    pub fn get(&self, t: &T) -> Option<Interned<'_, T>> {
        self.lock.with_ref(|_guard| {
            #[allow(unsafe_code)] // SAFETY: access is guarded by the lock
            let arena = unsafe { &*self.arena.get() };

            #[allow(unsafe_code)] // SAFETY: Interned ties the lifetime to the interner.
            arena
                .get_key_value(t)
                .map(|(t, _)| Interned(unsafe { t.as_ref() }))
        })
    }
}

impl<T: 'static + ?Sized, S: 'static, L: 'static + InternerLock<InternerState>> Interner<T, S, L> {
    /// Leak the owned [`Interner`] to make its references `'static`
    pub fn leak(self) -> LeakedInterner<T, S, L> {
        LeakedInterner::leak(self)
    }
}

#[cfg(feature = "raw")]
impl<T: ?Sized, S, L: InternerLock<InternerState>> Interner<T, S, L> {
    /// Raw interning interface for any `T`.
    pub fn intern_raw_mut<Q>(
        &mut self,
        it: Q,
        hash: u64,
        mut is_match: impl FnMut(&Q, &T) -> bool,
        do_hash: impl Fn(&T) -> u64,
        commit: impl FnOnce(Q) -> Box<T>,
    ) -> Interned<'_, T> {
        self.lock.with_mut_ref_mut(|_guard| {
            let arena = self.arena.get_mut();

            if let Some((t, _)) = arena.raw_entry().from_hash(hash, |t| is_match(&it, t)) {
                #[allow(unsafe_code)] // SAFETY: Interned ties the lifetime to the interner.
                return Interned(unsafe { t.as_ref() });
            }

            match arena.raw_entry_mut().from_hash(hash, |t| is_match(&it, t)) {
                #[allow(unsafe_code)] // SAFETY: Interned ties the lifetime to the interner.
                RawEntryMut::Occupied(entry) => Interned(unsafe { entry.key().as_ref() }),
                RawEntryMut::Vacant(entry) => {
                    let boxed = PinBox::new(commit(it));
                    #[allow(unsafe_code)] // SAFETY: Interned ties the lifetime to the interner.
                    let interned = Interned(unsafe { boxed.as_ref() });
                    entry.insert_with_hasher(hash, boxed, (), |t| do_hash(t));
                    interned
                },
            }
        })
    }

    /// Raw interning interface for any `T`.
    pub fn intern_raw<Q>(
        &self,
        it: Q,
        hash: u64,
        mut is_match: impl FnMut(&Q, &T) -> bool,
        do_hash: impl Fn(&T) -> u64,
        commit: impl FnOnce(Q) -> Box<T>,
    ) -> Interned<'_, T> {
        if let Some(interned) = self.get_raw(hash, |t| is_match(&it, t)) {
            return interned;
        }

        self.lock.with_ref_mut(|_guard| {
            #[allow(unsafe_code)] // SAFETY: access is guarded by the lock
            let arena = unsafe { &mut *self.arena.get() };

            match arena.raw_entry_mut().from_hash(hash, |t| is_match(&it, t)) {
                #[allow(unsafe_code)] // SAFETY: Interned ties the lifetime to the interner.
                RawEntryMut::Occupied(entry) => Interned(unsafe { entry.key().as_ref() }),
                RawEntryMut::Vacant(entry) => {
                    let boxed = PinBox::new(commit(it));
                    #[allow(unsafe_code)] // SAFETY: Interned ties the lifetime to the interner.
                    let interned = Interned(unsafe { boxed.as_ref() });
                    entry.insert_with_hasher(hash, boxed, (), |t| do_hash(t));
                    interned
                },
            }
        })
    }

    /// Raw interned reference lookup.
    pub fn get_raw(
        &self,
        hash: u64,
        mut is_match: impl FnMut(&T) -> bool,
    ) -> Option<Interned<'_, T>> {
        self.lock.with_ref(|_guard| {
            #[allow(unsafe_code)] // SAFETY: access is guarded by the lock
            let arena = unsafe { &*self.arena.get() };

            #[allow(unsafe_code)] // SAFETY: Interned ties the lifetime to the interner.
            arena
                .raw_entry()
                .from_hash(hash, |t| is_match(t))
                .map(|(t, _)| Interned(unsafe { t.as_ref() }))
        })
    }
}

#[cfg(feature = "std")]
impl<T: ?Sized, L: InternerLock<InternerState>> Interner<T, RandomState, L> {
    /// Create an empty interner.
    ///
    /// The backing set is initially created with a capacity of 0,
    /// so it will not allocate until it is first inserted into.
    pub fn new() -> Self {
        Interner {
            lock: L::new(()),
            arena: UnsafeCell::new(HashMap::default()),
        }
    }

    /// Create an empty interner with the specified capacity.
    ///
    /// The interner will be able to hold at least `capacity` items without reallocating.
    /// If `capacity` is 0, the interner will not initially allocate.
    pub fn with_capacity(capacity: usize) -> Self {
        Interner {
            lock: L::new(()),
            arena: UnsafeCell::new(HashMap::with_capacity_and_hasher(
                capacity,
                RandomState::default(),
            )),
        }
    }
}

/// Constructors to control the backing `HashSet`'s hash function
impl<T: ?Sized, H: BuildHasher, L: InternerLock<InternerState>> Interner<T, H, L> {
    /// Create an empty interner which will use the given hasher to hash the values.
    ///
    /// The interner is also created with the default capacity.
    pub const fn with_hasher(hasher: H) -> Self {
        Interner {
            lock: L::CONST_INIT,
            arena: UnsafeCell::new(HashMap::with_hasher(hasher)),
        }
    }

    /// Create an empty interner with the specified capacity, using `hasher` to hash the values.
    ///
    /// The interner will be able to hold at least `capacity` items without reallocating.
    /// If `capacity` is 0, the interner will not initially allocate.
    pub fn with_capacity_and_hasher(capacity: usize, hasher: H) -> Self {
        Interner {
            lock: L::new(()),
            arena: UnsafeCell::new(HashMap::with_capacity_and_hasher(capacity, hasher)),
        }
    }
}
