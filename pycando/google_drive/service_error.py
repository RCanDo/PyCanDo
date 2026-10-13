#! pip install google-api-python-client
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from googleapiclient.discovery import Resource, build


def get_gd_service(creds_json: str) -> Resource:
    """
    creds_json: str,
        path to JSON woth GoogleDrive credentials;
    Returns GoogleDrive `service: Resource`
    """
    creds = ServiceAccountCredentials.from_service_account_file(
        creds_json,
        scopes=['https://www.googleapis.com/auth/drive']
    )
    service = build('drive', 'v3', credentials=creds)
    return service


service = get_gd_service("dev/chat_partner/workprofi-exchange-area.json")
drives = service.drives().list().execute()['drives']
print(drives)


"""  !!!
HttpError: <HttpError 403 when requesting https://www.googleapis.com/drive/v3/drives?alt=json
returned "Google Drive API has not been used in project 835024038784 before or it is disabled.
Enable it by visiting https://console.developers.google.com/apis/api/drive.googleapis.com/overview?project=835024038784
then retry.
If you enabled this API recently, wait a few minutes for the action to propagate to our systems and retry.".
Details: "[{'message': 'Google Drive API has not been used in project 835024038784 before or it is disabled. Enable it by visiting https://console.developers.google.com/apis/api/drive.googleapis.com/overview?project=835024038784 then retry. If you enabled this API recently, wait a few minutes for the action to propagate to our systems and retry.', 'domain': 'usageLimits', 'reason': 'accessNotConfigured', 'extendedHelp': 'https://console.developers.google.com'}]">
"""
