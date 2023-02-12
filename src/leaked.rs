#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

use core::{
    borrow::Borrow,
    fmt,
    hash::{BuildHasher, Hash},
    marker::PhantomData,
};

use crate::{
    interned::Interned,
    interner::{Interner, InternerLock, InternerState},
};

/// A leaked [`Interner`], which provides `'static` references and is [`Send`]
/// even if the `&'static Interner` is not.
pub struct LeakedInterner<T: 'static + ?Sized, S: 'static, L: 'static + InternerLock<InternerState>>
{
    interner: &'static mut Interner<T, S, L>,
    marker: PhantomData<(&'static T, S, L)>,
}

#[allow(unsafe_code)]
unsafe impl<
        T: 'static + ?Sized + Send,
        S: 'static + Send,
        L: 'static + InternerLock<InternerState> + Send,
    > Send for LeakedInterner<T, S, L>
{
}

#[allow(unsafe_code)]
unsafe impl<
        T: 'static + ?Sized + Sync,
        S: 'static + Sync,
        L: 'static + InternerLock<InternerState> + Sync,
    > Sync for LeakedInterner<T, S, L>
{
}

impl<T: 'static + ?Sized + fmt::Debug, S: 'static, L: 'static + InternerLock<InternerState>>
    fmt::Debug for LeakedInterner<T, S, L>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LeakedInterner")
            .field("interner", self.interner)
            .finish()
    }
}

impl<T: 'static + ?Sized, S: 'static, L: 'static + InternerLock<InternerState>>
    LeakedInterner<T, S, L>
{
    /// Leaks an owned [`Interner`] to make its references `'static`
    pub fn leak(interner: Interner<T, S, L>) -> Self {
        Self {
            interner: Box::leak(Box::new(interner)),
            marker: PhantomData::<(&'static T, S, L)>,
        }
    }
}

impl<
        T: 'static + Eq + Hash + ?Sized,
        S: 'static + BuildHasher,
        L: 'static + InternerLock<InternerState>,
    > LeakedInterner<T, S, L>
{
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
    pub fn intern_mut<B>(&mut self, t: B) -> Interned<'static, T>
    where
        B: Borrow<T> + Into<Box<T>>,
    {
        #[allow(unsafe_code)] // SAFETY: the mutable reference is never exposed
        let interner: &'static mut Interner<T, S, L> =
            unsafe { &mut *(self.interner as *mut Interner<T, S, L>) };
        interner.intern_mut(t)
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
    pub fn intern<B>(&self, t: B) -> Interned<'static, T>
    where
        B: Borrow<T> + Into<Box<T>>,
    {
        #[allow(unsafe_code)] // SAFETY: the reference is never exposed
        let interner: &'static Interner<T, S, L> =
            unsafe { &*(self.interner as *const Interner<T, S, L>) };
        interner.intern(t)
    }

    /// Get an interned reference out of this interner.
    ///
    /// The returned reference is bound to the lifetime of the borrow used for
    /// this method. This guarantees that the returned reference will live no
    /// longer than this interner does.
    pub fn get(&self, t: &T) -> Option<Interned<'static, T>> {
        #[allow(unsafe_code)] // SAFETY: the reference is never exposed
        let interner: &'static Interner<T, S, L> =
            unsafe { &*(self.interner as *const Interner<T, S, L>) };
        interner.get(t)
    }
}

#[cfg(feature = "raw")]
impl<T: 'static + ?Sized, S: 'static, L: 'static + InternerLock<InternerState>>
    LeakedInterner<T, S, L>
{
    /// Raw interning interface for any `T`.
    pub fn intern_raw_mut<Q>(
        &mut self,
        it: Q,
        hash: u64,
        is_match: impl FnMut(&Q, &T) -> bool,
        do_hash: impl Fn(&T) -> u64,
        commit: impl FnOnce(Q) -> Box<T>,
    ) -> Interned<'static, T> {
        #[allow(unsafe_code)] // SAFETY: the mutable reference is never exposed
        let interner: &'static mut Interner<T, S, L> =
            unsafe { &mut *(self.interner as *mut Interner<T, S, L>) };
        interner.intern_raw_mut(it, hash, is_match, do_hash, commit)
    }

    /// Raw interning interface for any `T`.
    pub fn intern_raw<Q>(
        &self,
        it: Q,
        hash: u64,
        is_match: impl FnMut(&Q, &T) -> bool,
        do_hash: impl Fn(&T) -> u64,
        commit: impl FnOnce(Q) -> Box<T>,
    ) -> Interned<'static, T> {
        #[allow(unsafe_code)] // SAFETY: the reference is never exposed
        let interner: &'static Interner<T, S, L> =
            unsafe { &*(self.interner as *const Interner<T, S, L>) };
        interner.intern_raw(it, hash, is_match, do_hash, commit)
    }

    /// Raw interned reference lookup.
    pub fn get_raw(
        &self,
        hash: u64,
        is_match: impl FnMut(&T) -> bool,
    ) -> Option<Interned<'static, T>> {
        #[allow(unsafe_code)] // SAFETY: the reference is never exposed
        let interner: &'static Interner<T, S, L> =
            unsafe { &*(self.interner as *const Interner<T, S, L>) };
        interner.get_raw(hash, is_match)
    }
}
