#!/usr/bin/env python3
"""
title: GDrive upload files
remark:
    - original in
      ~/QuantUp/Savi/notes/whatsapp/
sources:
    - title: Google Drive API
      link: https://googleapis.github.io/google-api-python-client/docs/dyn/drive_v3.html
    - title: Google Drive API | drives
      link: https://googleapis.github.io/google-api-python-client/docs/dyn/drive_v3.drives.html
    - title: Google Drive API | files
      link: https://googleapis.github.io/google-api-python-client/docs/dyn/drive_v3.files.html
    - title: File-specific query terms
      link: https://developers.google.com/workspace/drive/api/guides/ref-search-terms#file-properties
    - title: Package googleapiclient
      link: https://googleapis.github.io/google-api-python-client/docs/epy/googleapiclient-module.html
    - title: " | Module http – Classes to encapsulate a single HTTP request.
      link: https://googleapis.github.io/google-api-python-client/docs/epy/googleapiclient.http-module.html
    - title: " | " | Class MediaIoBaseUpload
      link: https://googleapis.github.io/google-api-python-client/docs/epy/googleapiclient.http.MediaIoBaseUpload-class.html
"""
# %%
from google.oauth2.service_account import Credentials as ServiceAccountCredentials  # type: ignore
# from google.oauth2.credentials import Credentials as OAuthCredentials  # type: ignore   # alternative way
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload  # , MediaFileUpload
from io import BytesIO, BufferedReader

# %%  Step 0. Credentials
"""
Here is just one possible way of creating `google.oauth2.service_account.Credentials` object
– based on credentials stored in .json.
How to acquire this .json is another story, see [....].
"""
creds = ServiceAccountCredentials.from_service_account_file(
    'gdrive-test-creds.json',
    scopes=['https://www.googleapis.com/auth/drive']
)
service = build('drive', 'v3', credentials=creds)

drives = service.drives().list().execute()['drives']
drive_id = next(d['id'] for d in drives if d['name'] == "Test Area")


# %%  Chapter 1. Uploading file onto GD drive (not sub-folder)
# %%
"""
Uploading file onto GDrive is to `.create()` a new file with the content of local file.
It means that the process of copying is done low-level way, i.e.
- we must create `io.BytesIO` object from local file
  and this `io.BytesIO` are the body of the target file on the GD;
- only then `.create()` file on the GD with some name and body given via just mentioned `io.BytesIO` object.

Below we copy/create file into GDrive disk with given `drive_id` (acquired above).
To copy/create into specific subfolder one must first acquire id of this subfolder.
How to do it is shown in Chapter 2.
"""
# %%  Step 1
# %% Ex 1. we create `io.Bytes` on the fly
f = BytesIO("Test file contents".encode())
# hence we do not refere to any existing local file
file_name_local = None
# yet we need the target file name (i.e. the name of the file we .create() on GDrive)
file_name_gd = 'test.txt'
# and its mime-type
mime_type = 'text/plain'

# go to Step 2

# %%  Ex 2. from file on disk
# file_name_local = "Skad sie bierze woda sodowa.pdf"
# file_name_gd = "woda_sodowa.pdf"
# mime_type = 'application/pdf'
#
file_name_local = 'burning_man_unicorn.png'
file_name_gd = "unicorn.pdf"
mime_type = 'image/png'
#
# file_name_local = "fire-1080.jpg"
# file_name_gd = "fire.jpg"
file_name_local = "jesz_za_szybko.jpg"
file_name_gd = "jesz.jpg"
mime_type = 'image/jpg'
#
file_name_local = "c-dur.gif"
file_name_gd = "c-dur.gif"
mime_type = 'image/gif'
#
file_name_local = "tensor_calculus.pdf"
file_name_gd = "tensor_calculus.pdf"
mime_type = 'application/pdf'
#
file_name_local = "Mackinder.md"
file_name_gd = "Mackinder.md"
mime_type = "text/markdown"

# as shown later the following is unnecessary for "real" files (read from disk), because it's a kind of tautology;
# see Step 2';
# yet it is consistent with the basic logic behind the procedure (i.e. it's good as illustration):
f = BytesIO()
with open(file_name_local, 'rb') as g:
    f.write(g.read())
f.seek(0)
f.__sizeof__()  # just to see

# go to Step 2

# %%  Step 2
# %%  finally: create file on GD with the content of `io.BytesIO` object
def upload_to_gd(file: BytesIO | BufferedReader) -> dict[str, str]:
    file = service.files().create(
            body={
                'name': file_name_gd,
                'parents': [drive_id],   #  ['1N_GHO23bv-JbhcdaWAH8wJBj1n38_lpi']
            },
            media_body=MediaIoBaseUpload(
                file,                    # io.BytesIO / Python file object ~= io.BufferedReader
                mimetype=mime_type
            ),
            supportsAllDrives=True,
            # fields='name,webViewLink'   # alternative fields to be returned in response (instead of deafults)
        ).execute()
    return file

# %%
upload_to_gd(f)

# %%  Step 2'
"""
! However,
as states [here](https://googleapis.github.io/google-api-python-client/docs/epy/googleapiclient.http.MediaIoBaseUpload-class.html)
! Note that the Python file object is compatible with `io.Base` and can be used with this class also [i.e. directly].
Hence the following works too:
"""
upload_to_gd(open(file_name_local, 'rb'))

# btw:
isinstance(open(file_name_local, 'rb'), BufferedReader)  # True

# %%  Chapter 2. Acquiring id of files and folders
from googleapiclient.errors import HttpError


def get_files_info(file_type: str, drive_id: str) -> list[dict[str, str]] | None:
    """
    Search for files of the specific type (also folders) in drive location.
    https://googleapis.github.io/google-api-python-client/docs/dyn/drive_v3.files.html#list
    Notice tjat folders and files in the trash are also reported.
    Moreover, there may be many files and folders with the same name but different IDs.
    """
    match file_type:
        case 'folder':
            mime_type = 'application/vnd.google-apps.folder'
        case 'pdf':
            mime_type = 'application/pdf'
        case 'png' | 'gif' | 'jpg' | 'jpeg':
            mime_type = f'image/{file_type}'
        case 'txt' | 'md':
            mime_type = 'plain/text'
        case _:
            mime_type = 'application/json'  # whatever

    try:
        files = []
        page_token = None
        while True:
            response = (  # the clue
                service.files()
                .list(
                    corpora='drive',
                    q=f"mimeType='{mime_type}'",   # ! notice single quotes around {} !
                    spaces="drive",
                    fields="nextPageToken, files(id, name, createdTime)",
                    pageToken=page_token,
                    driveId=drive_id,
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True,
                )
                .execute()
            )
            for file in response.get("files", []):
                # Process change
                print(f'Found file: {file.get("name")}, {file.get("id")}')
            files.extend(response.get("files", []))
            page_token = response.get("nextPageToken", None)
            if page_token is None:
                break

    except HttpError as error:
      print(f"An error occurred: {error}")
      files = None

    return files

# %%
search_files('folder', drive_id)
search_files('png', drive_id)

# %%
"""
[{'id': '10ouWcwrefz6mgM8iv7lLC6l8HgnZNQ-L',  'name': 'fodder',  'createdTime': '2025-07-21T09:16:03.738Z'},
 {'id': '1dUAIY4dRgjvrM6G9y7IXDdQ41ziJR5QV',  'name': 'Savi',  'createdTime': '2025-07-21T08:13:33.745Z'},
 ...
 {'id': '1N_GHO23bv-JbhcdaWAH8wJBj1n38_lpi',  'name': 'Savi',  'createdTime': '2025-07-16T16:21:03.970Z'},
 {'id': '1joLmpPTcKqA1N6itKU1UUoR_lDiIpJQj',  'name': 'testowy-folder',  'createdTime': '2025-07-14T14:25:34.848Z'}]
"""

# %%
def create_folder(name: str, drive_id: str) -> str | None:
    """
    Create a folder and prints the folder ID
    Returns : Folder Id
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

# %%
create_folder("my_folder", drive_id)

# %%  delete files
# https://googleapis.github.io/google-api-python-client/docs/dyn/drive_v3.files.html#delete
response = service.files().delete(
    fileId='1KpYNAMc2kom6lrTZER9FaGX4CS9Yo_aN',
    supportsAllDrives=True,
).execute()

# %%
from googleapiclient.http import MediaFileUpload  # , MediaIoBaseUpload,
from os import stat


def upload_to_folder(
        file_name_local: str,
        file_name_gd: str,
        mime_type: str,
        folder_id: str,
        chunksize: int = 262144,
    ):
    """
    Upload a file to the specified folder and prints file ID, folder ID
    Returns: ID of the file uploaded

    Upload is done by chunks – it makes this method applicable to large files (over 5MB) as well.

    Basic building blocks:
    https://googleapis.github.io/google-api-python-client/docs/dyn/drive_v3.files.html#create
    https://googleapis.github.io/google-api-python-client/docs/epy/googleapiclient.http.MediaFileUpload-class.html
    """

    try:
        # create drive api client

        file_metadata = {"name": file_name_gd, "parents": [folder_id]}
        media = MediaFileUpload(
            file_name_local,
            mimetype=mime_type,
            resumable=True,     # !
            chunksize=262144    # bytes
        )
        # pylint: disable=maybe-no-member
        file = (  # the clue
            service.files()
            .create(
                body=file_metadata,
                media_body=media,
                supportsAllDrives=True,
                # fields='name,webViewLink,id'
            )
            # .execute()
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

# %%
file_name_local = 'burning_man_unicorn.png'  # local
file_name_gd = 'unicorn.png'
mime_type = 'image/png'
folder_id = '1N_GHO23bv-JbhcdaWAH8wJBj1n38_lpi'

upload_to_folder(
    file_name_local,
    file_name_gd,
    mime_type,
    folder_id='1N_GHO23bv-JbhcdaWAH8wJBj1n38_lpi',
)

# %%
