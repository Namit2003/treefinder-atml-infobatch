#!/usr/bin/env python3
"""
scripts/setup_gdrive.py
------------------------
One-time setup script to authorize the app with Google Drive and save token.json.

Run this ONCE on your remote server:
    python scripts/setup_gdrive.py

It will print a URL — open it in any browser (on your local machine), complete
the OAuth flow, then paste the authorization code back into the terminal.

After this, all training runs will automatically upload results to Google Drive.

Prerequisites:
    1. Go to https://console.cloud.google.com
    2. Create a project (or use existing)
    3. Enable the Google Drive API
    4. Go to APIs & Services → Credentials → Create Credentials → OAuth 2.0 Client ID
    5. Application type: Desktop app
    6. Download JSON → rename to credentials.json → put in project root
    7. Run this script
"""

import sys
import os
from pathlib import Path

# Allow running from project root or scripts/ dir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CREDENTIALS_PATH = "credentials.json"
TOKEN_PATH = "token.json"
SCOPES = ['https://www.googleapis.com/auth/drive.file']


def main():
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        print(
            "ERROR: Google API libraries not installed.\n"
            "Run:  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        )
        sys.exit(1)

    if not Path(CREDENTIALS_PATH).exists():
        print(
            f"ERROR: '{CREDENTIALS_PATH}' not found in current directory.\n\n"
            "Steps to get it:\n"
            "  1. Go to https://console.cloud.google.com\n"
            "  2. Select your project → APIs & Services → Credentials\n"
            "  3. Create OAuth 2.0 Client ID (Desktop app)\n"
            "  4. Download JSON and rename it to credentials.json\n"
            "  5. Upload credentials.json to your remote server (project root)\n"
            "  6. Re-run this script"
        )
        sys.exit(1)

    print("=" * 60)
    print("  Google Drive Authorization Setup")
    print("=" * 60)
    print()
    print("A URL will appear below. Open it in any browser (on your")
    print("local machine), sign in with your Google account, grant")
    print("the requested permissions, then copy-paste the code here.")
    print()

    flow = InstalledAppFlow.from_client_secrets_file(
        CREDENTIALS_PATH,
        SCOPES,
        redirect_uri="urn:ietf:wg:oauth:2.0:oob"  # out-of-band: no local server needed
    )
    auth_url, _ = flow.authorization_url(prompt='consent')

    print("Open this URL in your browser (on your local machine):")
    print()
    print(f"  {auth_url}")
    print()
    code = input("Paste the authorization code here: ").strip()
    flow.fetch_token(code=code)
    creds = flow.credentials

    # Save token
    with open(TOKEN_PATH, 'w') as f:
        f.write(creds.to_json())

    print()
    print(f"✓ token.json saved successfully.")
    print()
    print("Testing connection to Google Drive...")

    try:
        service = build('drive', 'v3', credentials=creds)
        about = service.about().get(fields="user").execute()
        user = about.get('user', {})
        print(f"✓ Connected as: {user.get('displayName', 'Unknown')} ({user.get('emailAddress', '?')})")
        print()
        print("Setup complete! Your training runs will now auto-upload to Google Drive.")
    except Exception as e:
        print(f"WARNING: Connection test failed: {e}")
        print("token.json was saved — upload may still work during training.")


if __name__ == "__main__":
    main()
