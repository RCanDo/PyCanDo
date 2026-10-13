#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:44:01 2026

@author: arek
"""
from gd_upload import get_gd_service
from gd_upload import upload_to_gd_folder_from_binary
from gd_upload import get_drive_id, get_files_info

GD_CREDS = "workprofi-exchange-area.json"
GD_FOLDER_ID = "1joLmpPTcKqA1N6itKU1UUoR_lDiIpJQj"


service = get_gd_service()
service   # <googleapiclient.discovery.Resource at 0x7041586119d0>

drives = service.drives().list().execute()['drives']
drives
# [{'kind': 'drive#drive',
#   'id': '0AEMhbyWUbU0HUk9PVA',
#   'name': 'WorkProfi Exchange Area'}]

drid = get_drive_id('WorkProfi Exchange Area', service)
drid   # '0AEMhbyWUbU0HUk9PVA'

files = get_files_info('folder', '0AEMhbyWUbU0HUk9PVA', service)
files
