import { initializeApp } from "https://www.gstatic.com/firebasejs/11.6.1/firebase-app.js";
import {
  createUserWithEmailAndPassword,
  getAuth,
  onAuthStateChanged,
  sendPasswordResetEmail,
  signInWithEmailAndPassword,
  signOut,
  updateProfile
} from "https://www.gstatic.com/firebasejs/11.6.1/firebase-auth.js";
import {
  doc,
  getDoc,
  getFirestore,
  serverTimestamp,
  setDoc
} from "https://www.gstatic.com/firebasejs/11.6.1/firebase-firestore.js";
import firebaseConfig from "./firebase-config.js";

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);

const authGate = document.getElementById("authGate");
const authForm = document.getElementById("authForm");
const authNameField = document.getElementById("authNameField");
const authName = document.getElementById("authName");
const authEmail = document.getElementById("authEmail");
const authPassword = document.getElementById("authPassword");
const authTitle = document.getElementById("authTitle");
const authDescription = document.getElementById("authDescription");
const authSubmit = document.getElementById("authSubmit");
const authError = document.getElementById("authError");
const authModeToggle = document.getElementById("authModeToggle");
const authResetPassword = document.getElementById("authResetPassword");
const accountButton = document.getElementById("accountButton");
const accountName = document.getElementById("accountName");
const accountAvatar = document.getElementById("accountAvatar");
const profileBackdrop = document.getElementById("profileBackdrop");
const profileForm = document.getElementById("profileForm");
const profileEmail = document.getElementById("profileEmail");
const profileError = document.getElementById("profileError");
const profileSave = document.getElementById("profileSave");

let isSignUp = false;
let currentUser = null;
let savedProfile = null;
let pendingSignupName = "";

function authErrorMessage(error) {
  const messages = {
    "auth/email-already-in-use": "An account already uses this email. Sign in instead.",
    "auth/invalid-credential": "Email or password is incorrect.",
    "auth/invalid-email": "Enter a valid email address.",
    "auth/network-request-failed": "Network error. Check your connection and try again.",
    "auth/operation-not-allowed": "Email and password sign-in is not enabled for this Firebase project.",
    "auth/too-many-requests": "Too many attempts. Wait a little and try again.",
    "auth/weak-password": "Choose a stronger password with at least 8 characters.",
    "auth/user-disabled": "This account is disabled. Contact the project administrator."
  };
  return messages[error.code] || "Sign-in could not be completed. Please try again.";
}

function setFormBusy(form, button, busy, idleLabel) {
  for (const control of form.querySelectorAll("button, input, select")) {
    control.disabled = busy;
  }
  button.textContent = busy ? "PLEASE WAIT…" : idleLabel;
}

function setAuthMode(signUp) {
  isSignUp = signUp;
  authNameField.hidden = !signUp;
  authName.required = signUp;
  authPassword.autocomplete = signUp ? "new-password" : "current-password";
  authTitle.textContent = signUp ? "Create your account" : "Welcome back";
  authDescription.textContent = signUp
    ? "Create a secure account to set up your personal dashboard."
    : "Sign in to open your biometric dashboard and manage your profile.";
  authSubmit.textContent = signUp ? "CREATE ACCOUNT" : "SIGN IN";
  authModeToggle.textContent = signUp ? "ALREADY REGISTERED? SIGN IN" : "NEW HERE? CREATE ACCOUNT";
  authResetPassword.hidden = signUp;
  authError.textContent = "";
}

function setProfileFields(profile, user) {
  document.getElementById("profileName").value = profile?.name || user.displayName || pendingSignupName || "";
  document.getElementById("profileAge").value = profile?.age ?? "";
  document.getElementById("profileGender").value = profile?.gender || "";
  document.getElementById("profileHeight").value = profile?.heightCm ?? "";
  document.getElementById("profileWeight").value = profile?.weightKg ?? "";
  profileEmail.textContent = user.email || "";
}

function openProfile() {
  if (!currentUser) return;
  setProfileFields(savedProfile, currentUser);
  profileError.textContent = "";
  profileBackdrop.hidden = false;
  document.getElementById("profileName").focus();
}

function showProfileButton(user, profile) {
  const displayName = profile?.name || user.displayName || user.email || "My profile";
  accountName.textContent = displayName;
  accountAvatar.textContent = displayName.trim().charAt(0).toUpperCase() || "U";
  accountButton.hidden = false;
}

function validateProfileNumber(value, label, min, max) {
  if (value === "") return null;
  const number = Number(value);
  if (!Number.isFinite(number) || number < min || number > max) {
    throw new Error(`${label} must be between ${min} and ${max}.`);
  }
  return number;
}

authModeToggle.addEventListener("click", () => setAuthMode(!isSignUp));

authForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  authError.textContent = "";
  const email = authEmail.value.trim();
  const password = authPassword.value;
  const name = authName.value.trim();
  if (isSignUp && !name) {
    authError.textContent = "Enter your name to create an account.";
    authName.focus();
    return;
  }
  if (isSignUp && password.length < 8) {
    authError.textContent = "Use a password with at least 8 characters.";
    authPassword.focus();
    return;
  }

  const idleLabel = isSignUp ? "CREATE ACCOUNT" : "SIGN IN";
  setFormBusy(authForm, authSubmit, true, idleLabel);
  try {
    if (isSignUp) {
      pendingSignupName = name;
      const credential = await createUserWithEmailAndPassword(auth, email, password);
      await updateProfile(credential.user, { displayName: name });
    } else {
      await signInWithEmailAndPassword(auth, email, password);
    }
  } catch (error) {
    if (!auth.currentUser) pendingSignupName = "";
    console.error("Firebase Authentication failed:", error);
    authError.textContent = authErrorMessage(error);
  } finally {
    setFormBusy(authForm, authSubmit, false, idleLabel);
  }
});

authResetPassword.addEventListener("click", async () => {
  authError.textContent = "";
  const email = authEmail.value.trim();
  if (!email) {
    authError.textContent = "Enter your email address first.";
    authEmail.focus();
    return;
  }
  authResetPassword.disabled = true;
  try {
    await sendPasswordResetEmail(auth, email);
    authError.textContent = "If an account exists for this address, a password reset email has been sent.";
  } catch (error) {
    console.error("Firebase password reset failed:", error);
    authError.textContent = authErrorMessage(error);
  } finally {
    authResetPassword.disabled = false;
  }
});

document.getElementById("profileClose").addEventListener("click", () => {
  profileBackdrop.hidden = true;
});

profileBackdrop.addEventListener("click", (event) => {
  if (event.target === profileBackdrop) profileBackdrop.hidden = true;
});

document.getElementById("profileSignOut").addEventListener("click", async () => {
  try {
    await signOut(auth);
  } catch (error) {
    console.error("Firebase sign-out failed:", error);
    profileError.textContent = "Could not sign out. Please try again.";
  }
});

profileForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  profileError.textContent = "";
  const name = document.getElementById("profileName").value.trim();
  if (!name) {
    profileError.textContent = "Enter your name.";
    document.getElementById("profileName").focus();
    return;
  }

  let profile;
  try {
    profile = { name };
    const age = validateProfileNumber(document.getElementById("profileAge").value, "Age", 1, 120);
    const heightCm = validateProfileNumber(document.getElementById("profileHeight").value, "Height", 30, 260);
    const weightKg = validateProfileNumber(document.getElementById("profileWeight").value, "Weight", 2, 500);
    const gender = document.getElementById("profileGender").value;
    if (age !== null) profile.age = Math.round(age);
    if (gender) profile.gender = gender;
    if (heightCm !== null) profile.heightCm = heightCm;
    if (weightKg !== null) profile.weightKg = weightKg;
  } catch (error) {
    profileError.textContent = error.message;
    return;
  }

  setFormBusy(profileForm, profileSave, true, "SAVE USER DETAILS");
  try {
    await setDoc(doc(db, "users", currentUser.uid), {
      ...profile,
      updatedAt: serverTimestamp()
    });
    await updateProfile(currentUser, { displayName: name });
    savedProfile = profile;
    showProfileButton(currentUser, savedProfile);
    profileBackdrop.hidden = true;
  } catch (error) {
    console.error("Saving Firebase user profile failed:", error);
    profileError.textContent = error.code === "permission-denied"
      ? "Profile access was denied. Check the deployed Firestore security rules."
      : "Could not save your profile. Check your connection and try again.";
  } finally {
    setFormBusy(profileForm, profileSave, false, "SAVE USER DETAILS");
  }
});

onAuthStateChanged(auth, async (user) => {
  currentUser = user;
  if (!user) {
    savedProfile = null;
    pendingSignupName = "";
    accountButton.hidden = true;
    profileBackdrop.hidden = true;
    authGate.hidden = false;
    authError.textContent = "";
    return;
  }

  authGate.hidden = true;
  profileEmail.textContent = user.email || "";
  try {
    const snapshot = await getDoc(doc(db, "users", user.uid));
    savedProfile = snapshot.exists() ? snapshot.data() : null;
    showProfileButton(user, savedProfile);
    if (!savedProfile) {
      openProfile();
      profileError.textContent = "Add your details to finish setting up your dashboard profile.";
    }
    pendingSignupName = "";
  } catch (error) {
    console.error("Loading Firebase user profile failed:", error);
    showProfileButton(user, null);
    openProfile();
    profileError.textContent = "Could not load saved profile details. Verify Firestore is configured and try again.";
  }
});

window.pulseHaloAuth = { openProfile };
