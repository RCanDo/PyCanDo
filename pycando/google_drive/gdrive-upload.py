#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sources:
    - link: https://stackoverflow.com/questions/60182791/how-to-upload-a-file-to-google-drive-using-python-and-the-drive-api-v3
remark:
    - original in
      ~/QuantUp/Savi/notes/whatsapp/
file:
    date: 2025-07-15
    authors:
        - arek
"""
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload  # , MediaIoBaseUpload,
# from io import BytesIO

# %%
from os import stat

def getFileByteSize(filename):
    # Get file size in python
    file_stats = stat(filename)
    print('File Size in Bytes is {}'.format(file_stats.st_size))
    return file_stats.st_size

# %%

creds = Credentials.from_service_account_file(
    'gdrive-test-creds.json',
    scopes=['https://www.googleapis.com/auth/drive']
)
service = build('drive', 'v3', credentials=creds)

drives = service.drives().list().execute()['drives']
drive_id = next(d['id'] for d in drives if d['name'] == "Test Area")

file_name_local = 'burning_man_unicorn.png'  # local
file_name_gd = 'unicorn.png'
mime_type = 'image/png'
resumable = True
chunk_size = 262144

files = service.files()

request = files.create(
    body={
        'name': file_name_gd,   # on GD
        'parents': [drive_id]
    },
    media_body=MediaFileUpload(
        filename=file_name_local,
        mimetype=mime_type,
        resumable=resumable,
        chunksize=chunk_size
    )
).execute()

if getFileByteSize(file_name_local) > chunk_size:
    response = None
    while response is None:
        chunk = request.next_chunk()
        if chunk:
            status, response = chunk
            if status:
                print("Uploaded %d%%." % int(status.progress() * 100))
print("Upload Complete!")

# %%
