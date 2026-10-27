# Firebase Hosting (no billing account)

The dashboard is hosted on the Firebase Spark plan as a static site. Email and
password authentication uses Firebase Authentication, while user profile
details are stored in the project's Firestore database in Delhi (`asia-south2`).
Firestore rules limit each account to reading and changing its own profile.
The Python API is not deployed; interactive analysis runs in the browser.

To build and deploy the live Firebase Hosting site in one step, run:

```sh
python3 deploy.py
```

This rebuilds `public/`, deploys Hosting to the default Firebase project in
`.firebaserc`, checks that the live site responds, and then prints the permanent
Hosting URL. The live `web.app` URL is not a temporary preview and has no
preview-channel expiration date. It remains available while the Firebase
project and Hosting site remain active and within Firebase's applicable plan
limits; no hosted service can be guaranteed to run forever.

The command requires the Firebase CLI and an authenticated Firebase account;
sign in once with `firebase login`. For a temporary preview channel instead,
run:

```sh
python3 build_standalone.py
firebase hosting:channel:deploy preview --project pulsehalo-smartring-14a47
```

Preview URLs expire after seven days by default. Running `python3 server.py`
starts only the local Python API; it does not deploy Hosting or create a preview
URL.

Enable Email/Password in Firebase Console under Authentication → Sign-in
method. The app's Firebase web configuration is public client configuration;
Firestore security rules, not the web API key, restrict profile access.

The generated site and client modules are in `public/`. Rebuild after changing
the dashboard, authentication client, model engine, or preset data. For local
API development, run `python3 server.py`; the deployed static dashboard does
not use it.
