#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sources:
    - link: https://proxiesapi.com/articles/downloading-binary-files-with-python-requests
file:
    date: 2026-05-05
    remark:

        - the same file (almost) as
          ~/QuantUp/Savi/savi/backend/onyx/connectors/whatsapp/files_zzz.py
        - see also some files in:
          ~/QuantUp/Savi/notes/whatsapp/        !!!
    authors:
        - arkadiusz.kasprzyk@quantup.pl
"""
import io
import requests
from os import stat
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
from googleapiclient.errors import HttpError
from googleapiclient.discovery import Resource, build  #
#
from google.oauth2.credentials import Credentials as OAuthCredentials  # type: ignore
from google.oauth2.service_account import Credentials as ServiceAccountCredentials  # type: ignore

GD_CREDS = "workprofi-exchange-area.json"
GD_FOLDER_ID = "1joLmpPTcKqA1N6itKU1UUoR_lDiIpJQj"


def get_gd_service(creds_json: str = GD_CREDS) -> Resource:
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


def binary_from_stream(stream_response: requests.Response, chunk_size: int = 1024) -> io.BytesIO:
    """
    stream_response: requests.Response,
        object created by `requests.get(..., stream=True)`
        thus having method `.iter_content()`;
    chunk_size: int = 1024,
        the only parameter passed to `stream_response.iter_content()`;
    """
    f = io.BytesIO()
    for chunk in stream_response.iter_content(chunk_size):
        f.write(chunk)
    f.seek(0)
    return f


def upload_to_gd_folder_from_binary(
        file_binary: io.BytesIO,
        file_name_gd: str,
        mime_type: str,
        folder_id: str,
        service: Resource,
        chunksize: int = 1024 ** 2,
    ) -> str | None:
    """
    Upload a binary object `file_binary` in memory to the GDrive folder specified by `folder_id`
    as a file with name `file_name_gd` and returns resulting GDrive file ID;

    Upload is done by chunks – it makes this method applicable to large files (over 5MB) as well.

    Basic building blocks:
    https://googleapis.github.io/google-api-python-client/docs/dyn/drive_v3.files.html#create
    https://googleapis.github.io/google-api-python-client/docs/epy/googleapiclient.http.MediaIoBaseUpload-class.html

    Arguments
    ---------
    file_binary: io.BinaryIO,
        handler to the file in binary form (in memory)
    file_name_gd: str,
        file name on the Google Drive
    mime_type: str,
        e.g. 'application/pdf', 'image/png', 'video/mpg4', 'text/plain', etc.
    folder_id: str,
    service: Resource,
        e.g.
        `service = get_gd_service('gdrive-creds.json')`.
    chunksize: int = 1024 * 1024,
        the actual value (in bytes) MUST be multiplicity of 262144 = 256 * 1024;
    """

    try:
        file_metadata = {"name": file_name_gd, "parents": [folder_id]}
        media = MediaIoBaseUpload(
            file_binary,
            mimetype=mime_type,
            resumable=True,
            chunksize=chunksize
        )
        file = (
            service.files()
            .create(
                body=file_metadata,
                media_body=media,
                supportsAllDrives=True,
                # fields='name,webViewLink,id'   # alternative fields to be returned in response
            )
        )

        if file_binary.__sizeof__() > chunksize:
            response = None
            while response is None:
                if chunk := file.next_chunk():
                    status, response = chunk
                    if status:
                        print("Uploaded %d%%." % int(status.progress() * 100))
        else:
            response = file.execute()
        print("Upload Complete!")

        file_id = response.get("id")
        print(f'File ID: "{file_id}".')
        return file_id

    except HttpError as error:
        print(f"An error occurred: {error}")
        return None


def upload_to_gd_folder_from_disk(
        file_name_local: str,
        file_name_gd: str,
        mime_type: str,
        folder_id: str,
        service: Resource,
        chunksize: int = 262144,
    ) -> str | None:
    """
    Upload a file to the specified folder and prints file ID, folder ID
    Returns: ID of the file uploaded

    Upload is done by chunks – it makes this method applicable to large files (over 5MB) as well.

    Basic building blocks:
    https://googleapis.github.io/google-api-python-client/docs/dyn/drive_v3.files.html#create
    https://googleapis.github.io/google-api-python-client/docs/epy/googleapiclient.http.MediaFileUpload-class.html

    Arguments
    ---------
    file_name_local: str,
        file name on the disk
    file_name_gd: str,
        file name on the Google Drive
    mime_type: str,
        e.g. 'application/pdf', 'image/png', 'video/mpg4', 'text/plain', etc.
    folder_id: str,
    service: Resource,
        e.g.
        `service = googleapiclient.discovery.build('drive', 'v3', credentials=creds)`
        where
        `creds = google.oauth2.service_account.Credentials.from_service_account_file(
            'gdrive-creds.json',
            scopes=['https://www.googleapis.com/auth/drive']
        )`
        see... [how to get GD-credentials in .json].
    chunksize: int = 262144,
        the actual value (in bytes) MUST be multiplicity of the default;
    """

    try:
        file_metadata = {"name": file_name_gd, "parents": [folder_id]}
        media = MediaFileUpload(
            file_name_local,
            mimetype=mime_type,
            resumable=True,
            chunksize=chunksize
        )
        file = (
            service.files()
            .create(
                body=file_metadata,
                media_body=media,
                supportsAllDrives=True,
                # fields='name,webViewLink,id'   # alternative fields to be returned in response
            )
        )

        if stat(file_name_local).st_size > chunksize:
            response = None
            while response is None:
                if chunk := file.next_chunk():
                    status, response = chunk
                    if status:
                        print("Uploaded %d%%." % int(status.progress() * 100))
        else:
            response = file.execute()
        print("Upload Complete!")

        file_id = response.get("id")
        print(f'File ID: "{file_id}".')
        return file_id

    except HttpError as error:
        print(f"An error occurred: {error}")
        return None


# %%  other helpers – not used in connector but may come in handy when testing
def get_drive_id(name: str, service: Resource) -> str:
    """
    Get ID of a drive with sepcific `name`.
    name: str,
        name of drive to search for its ID
    service: Resource,
        see help for `upload_to_gd_folder_from_disk()` above;
    """
    drives = service.drives().list().execute()['drives']
    drive_id = next(d['id'] for d in drives if d['name'] == name)
    return drive_id


def get_files_info(
        file_type: str,  # or mime_type
        drive_id: str,
        service: Resource,
        file_info: str = 'id, name, createdTime, parents, trashed',
        trashed: bool = False,
) -> list[dict[str, str]] | None:
    """
    Search for files of the specific type (also folders) in drive with given `drive_id`.
    Notice that folders and files in the trash are also reported. (If parent is trashed then file is also trashed.)
    Moreover, there may be many files or folders with the same name but different IDs.
    Thus this function returns by default 'id', 'name', 'createdTime', 'parents' and 'trashed' (if `trashed = True`).
    Hence to have readable overview of the drive state run:
    1.  get_files_info('folder', drive_id=drive_id, service=service)   # what folders there are and what are their IDs
    2.  get_files_info('jpg', drive_id=drive_id, service=service)      # file type must be specified

    Basic building block:
    https://googleapis.github.io/google-api-python-client/docs/dyn/drive_v3.files.html#list
    See also
    https://developers.google.com/workspace/drive/api/guides/ref-search-terms#file-properties
    for how to construct the query – `q` param in `service.files().list()`.

    file_type: str | None,
        currently only 'folder', 'pdf', 'png', 'gif', 'jpg', 'jpeg', 'txt', 'md';
        it's also possible to pass here arbitrary files' mime_type (which is recognised by `'/' in file_type`)
        however "arbitrary" means "any rocognisable by http request";
    drive_id: str,
        id of the drive to be scanned through in search for files or folders;
    service: Resource,
        see help for `upload_to_gd_folder_from_disk()` above;
    file_info: str = 'id, name, createdTime, parents, trashed',
        file infos to get; see the link above for the list of all possible fields;
    trashed: bool = False,
        if True then also trashed files will be shown;
    """

    if not '/' in file_type:
        match file_type:
            case 'folder':
                mime_type = 'application/vnd.google-apps.folder'
            case 'pdf':
                mime_type = 'application/pdf'
            case 'png' | 'gif':
                mime_type = f'image/{file_type}'
            case 'jpg' | 'jpeg':
                mime_type = 'image/jpeg'
            case 'txt' | 'text' | 'plain':
                mime_type = 'text/plain'
            case 'md' | 'markdown':
                mime_type = "text/markdown"
            case _:
                mime_type = 'application/json'  # whatever
    else:
        mime_type = file_type

    query = f"mimeType = '{mime_type}'"   # ! notice single quotes around {} !
    if not trashed:
        query += " and trashed = false"

    try:
        files = []
        page_token = None
        while True:
            response = (  # the clue
                service.files()
                .list(
                    corpora='drive',
                    q=query,
                    spaces="drive",
                    fields=f"nextPageToken, files({file_info})",
                    pageToken=page_token,
                    driveId=drive_id,
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True,
                )
                .execute()
            )
            files.extend(response.get("files", []))
            page_token = response.get("nextPageToken", None)
            if page_token is None:
                break


    except HttpError as error:
        print(f"An error occurred: {error}")
        files = None

    return files


def create_folder(name: str, drive_id: str, service: Resource) -> str | None:
    """
    Creates a folder with given `name` in the drive with `drive_id` and prints the folder id.
    Returns : Folder id
    https://googleapis.github.io/google-api-python-client/docs/dyn/drive_v3.files.html#create
    """

    try:
        file_metadata = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [drive_id]
        }

        file = service.files().create(  # the clue
            body=file_metadata,
            #fields="id"
            supportsAllDrives=True,
            fields='name,webViewLink,id'
        ).execute()
        print(f'Folder ID: "{file.get("id")}".')
        return file.get("id")

    except HttpError as error:
        print(f"An error occurred: {error}")
        return None


def delete_file(file_id: str, service: Resource) -> None:
    """
    doesn't work! why???
    # https://googleapis.github.io/google-api-python-client/docs/dyn/drive_v3.files.html#delete
    """
    response = service.files().delete(
        fileId=file_id,
        supportsAllDrives=True
    ).execute()
