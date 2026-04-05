/**
 * auth.js — Shared authentication module
 * Used by: landing.html, index.html, weather.html, crop.html, report.html
 *
 * Requires the page to have:
 *   #authModal, #authForm, #authFeedback, #authName, #authEmail,
 *   #authPassword, #authConfirm, #authSubmitBtn, #authCloseBtn,
 *   #nameGroup, #confirmGroup, #toggleLink, #signinLink,
 *   #authBtn, #signOutBtn, #userInfo, #userName
 */

// ── Storage helpers ───────────────────────────────────────────────────
function getUsers()     { return JSON.parse(localStorage.getItem('wp_users')   || '{}'); }
function saveUsers(u)   { localStorage.setItem('wp_users',   JSON.stringify(u)); }
function getSession()   { return JSON.parse(localStorage.getItem('wp_session') || 'null'); }
function saveSession(u) { localStorage.setItem('wp_session', JSON.stringify(u)); }
function clearSession() { localStorage.removeItem('wp_session'); }

// ── Feedback helpers ──────────────────────────────────────────────────
function showAuthFeedback(msg, type) {
    const el = document.getElementById('authFeedback');
    el.textContent = msg;
    el.className   = 'auth-feedback ' + type;
    el.classList.remove('hidden');
}
function clearAuthFeedback() {
    const el = document.getElementById('authFeedback');
    el.className   = 'auth-feedback hidden';
    el.textContent = '';
}
function markInputError(id)  { document.getElementById(id).classList.add('input-error'); }
function clearInputErrors()  { document.querySelectorAll('.input-error').forEach(el => el.classList.remove('input-error')); }

// ── Modal open / close ────────────────────────────────────────────────
function openAuthModal() {
    clearAuthFeedback();
    clearInputErrors();
    document.getElementById('authForm').reset();
    document.getElementById('authModal').classList.add('show');
}
function closeAuthModal() {
    document.getElementById('authModal').classList.remove('show');
    clearAuthFeedback();
    clearInputErrors();
}

// ── Tab switching ─────────────────────────────────────────────────────
let currentAuthMode = 'signin';

function switchAuthMode(e) {
    const mode = e.target.dataset.mode;
    currentAuthMode = mode;
    clearAuthFeedback();
    clearInputErrors();

    document.querySelectorAll('.modal-tab').forEach(t => t.classList.remove('active'));
    e.target.classList.add('active');

    const isSignup = mode === 'signup';
    document.getElementById('nameGroup').classList.toggle('hidden',    !isSignup);
    document.getElementById('confirmGroup').classList.toggle('hidden', !isSignup);
    document.getElementById('toggleLink').classList.toggle('hidden',    isSignup);
    document.getElementById('signinLink').classList.toggle('hidden',   !isSignup);
    document.getElementById('authSubmitBtn').textContent = isSignup ? 'Sign Up' : 'Sign In';
}

function toggleAuthMode(e)         { e.preventDefault(); document.querySelector('[data-mode="signup"]').click(); }
function toggleAuthModeToSignin(e) { e.preventDefault(); document.querySelector('[data-mode="signin"]').click(); }

// ── Sign up ───────────────────────────────────────────────────────────
function handleSignUp(name, email, password, confirm) {
    clearInputErrors();
    if (!name.trim()) {
        markInputError('authName');
        showAuthFeedback('Please enter your full name.', 'error'); return;
    }
    if (!email.trim() || !/^[^@]+@[^@]+\.[^@]+$/.test(email)) {
        markInputError('authEmail');
        showAuthFeedback('Please enter a valid email address.', 'error'); return;
    }
    if (password.length < 6) {
        markInputError('authPassword');
        showAuthFeedback('Password must be at least 6 characters.', 'error'); return;
    }
    if (password !== confirm) {
        markInputError('authPassword');
        markInputError('authConfirm');
        showAuthFeedback('Passwords do not match.', 'error'); return;
    }
    const users = getUsers();
    if (users[email.toLowerCase()]) {
        markInputError('authEmail');
        showAuthFeedback('An account with this email already exists. Please sign in.', 'error'); return;
    }
    users[email.toLowerCase()] = { name: name.trim(), email: email.toLowerCase(), password };
    saveUsers(users);
    showAuthFeedback('Account created! Signing you in…', 'success');
    setTimeout(() => applySignedInState({ name: name.trim(), email: email.toLowerCase() }), 900);
}

// ── Sign in ───────────────────────────────────────────────────────────
function handleSignIn(email, password) {
    clearInputErrors();
    if (!email.trim()) {
        markInputError('authEmail');
        showAuthFeedback('Please enter your email address.', 'error'); return;
    }
    if (!password) {
        markInputError('authPassword');
        showAuthFeedback('Please enter your password.', 'error'); return;
    }
    const users = getUsers();
    const user  = users[email.toLowerCase()];
    if (!user) {
        markInputError('authEmail');
        showAuthFeedback('No account found with this email. Please sign up.', 'error');
        document.getElementById('toggleLink').classList.remove('hidden'); return;
    }
    if (user.password !== password) {
        markInputError('authPassword');
        showAuthFeedback('Incorrect password. Please try again.', 'error'); return;
    }
    showAuthFeedback('Welcome back, ' + user.name.split(' ')[0] + '!', 'success');
    setTimeout(() => applySignedInState({ name: user.name, email: user.email }), 700);
}

// ── Apply signed-in UI state ──────────────────────────────────────────
// onSuccess: optional callback after sign-in (used by landing page for redirect)
function applySignedInState(user, onSuccess) {
    saveSession(user);

    const firstNameEl = document.getElementById('userName');
    const userInfoEl  = document.getElementById('userInfo');
    const authBtnEl   = document.getElementById('authBtn');
    const signOutEl   = document.getElementById('signOutBtn');

    if (firstNameEl)  firstNameEl.textContent  = user.name.split(' ')[0];
    if (userInfoEl)   userInfoEl.style.display  = 'block';
    if (authBtnEl)    authBtnEl.style.display   = 'none';
    if (signOutEl)    signOutEl.style.display   = 'block';

    closeAuthModal();

    if (typeof onSuccess === 'function') {
        onSuccess(user);
    }
}

// ── Sign out ──────────────────────────────────────────────────────────
function handleSignOut() {
    clearSession();

    const userInfoEl = document.getElementById('userInfo');
    const authBtnEl  = document.getElementById('authBtn');
    const signOutEl  = document.getElementById('signOutBtn');

    if (userInfoEl) userInfoEl.style.display = 'none';
    if (authBtnEl)  authBtnEl.style.display  = 'block';
    if (signOutEl)  signOutEl.style.display  = 'none';
}

// ── Form submit router ────────────────────────────────────────────────
function handleAuth(e) {
    e.preventDefault();
    const name     = document.getElementById('authName').value.trim();
    const email    = document.getElementById('authEmail').value.trim();
    const password = document.getElementById('authPassword').value;
    const confirm  = document.getElementById('authConfirm').value;
    if (currentAuthMode === 'signup') handleSignUp(name, email, password, confirm);
    else                              handleSignIn(email, password);
}

// ── Restore session on page load ──────────────────────────────────────
function restoreSession(onRestored) {
    const session = getSession();
    if (session) applySignedInState(session, onRestored);
}

// ── Wire up auth modal events (call once per page after DOM ready) ────
function initAuth(onSignInSuccess) {
    const authBtn    = document.getElementById('authBtn');
    const closeBtn   = document.getElementById('authCloseBtn');
    const signOutBtn = document.getElementById('signOutBtn');
    const form       = document.getElementById('authForm');
    const tabs       = document.querySelectorAll('.modal-tab');

    if (authBtn)    authBtn.addEventListener('click', openAuthModal);
    if (closeBtn)   closeBtn.addEventListener('click', closeAuthModal);
    if (signOutBtn) signOutBtn.addEventListener('click', handleSignOut);
    if (form)       form.addEventListener('submit', handleAuth);
    tabs.forEach(t => t.addEventListener('click', switchAuthMode));

    // Store callback so applySignedInState can call it
    if (typeof onSignInSuccess === 'function') {
        window.__authOnSuccess = onSignInSuccess;
    }
}

// Patch applySignedInState to call page-level callback if set
const _applySignedInState = applySignedInState;
// eslint-disable-next-line no-func-assign
applySignedInState = function(user, onSuccess) {
    _applySignedInState(user, onSuccess || window.__authOnSuccess);
};
