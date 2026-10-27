# Firebase Hosting (no billing account)

The dashboard is hosted on the Firebase Spark plan as a static site. Email and
password authentication uses Firebase Authentication, while user profile
details are stored in the project's Firestore database in Delhi (`asia-south2`).
Firestore rules limit each account to reading and changing its own profile.
The Python API is not deployed; interactive analysis runs in the browser.

Build the static site and deploy Hosting plus Firestore security rules:

```sh
python3 build_standalone.py
firebase deploy --only hosting,firestore --project pulsehalo-smartring-14a47
```

Enable Email/Password in Firebase Console under Authentication → Sign-in
method. The app's Firebase web configuration is public client configuration;
Firestore security rules, not the web API key, restrict profile access.

The generated site and client modules are in `public/`. Rebuild after changing
the dashboard, authentication client, model engine, or preset data. For local
API development, run `python3 server.py`; the deployed static dashboard does
not use it.
