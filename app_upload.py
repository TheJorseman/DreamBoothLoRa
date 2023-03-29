#!/usr/bin/env python

from __future__ import annotations

import pathlib

import slugify

from constants import UploadTarget
from uploader import Uploader


class LoRAModelUploader(Uploader):
    def upload_lora_model(
        self,
        folder_path: str,
        repo_name: str,
        upload_to: str,
        private: bool,
        delete_existing_repo: bool,
    ) -> str:
        if not folder_path:
            raise ValueError
        if not repo_name:
            repo_name = pathlib.Path(folder_path).name
        repo_name = slugify.slugify(repo_name)

        if upload_to == UploadTarget.PERSONAL_PROFILE.value:
            organization = ''
        elif upload_to == UploadTarget.LORA_LIBRARY.value:
            organization = 'lora-library'
        else:
            raise ValueError

        return self.upload(folder_path,
                           repo_name,
                           organization=organization,
                           private=private,
                           delete_existing_repo=delete_existing_repo)

