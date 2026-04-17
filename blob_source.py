import os

import requests

from mywai_python_integration_kit.apis import ApiObjectsContainer
from mywai_python_integration_kit.utils.logging_lib import get_mywai_logger

logger = get_mywai_logger()


def get_file_in_bytes_from_blob(path):
    base_url = ApiObjectsContainer.http_client.base_url
    api_url = (
        ApiObjectsContainer.api_filter.Storage.get_getBlob_with_blobPath.route.replace(
            "{blobPath}", path
        )
    )
    url = base_url + api_url
    logger.info(url)
    headers = ApiObjectsContainer.http_client.headers

    try:
        response = requests.get(url, headers=headers, verify=False)

        if response.status_code == 200:
            binary_data = response.content

            return binary_data
        else:
            logger.error(f"Error while downloading the file: {response.status_code}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error while making the request: {e}")


def download_file_from_blob(blob_path, folder_path: str = "") -> None:
    """
    Downloads an object from Blob Storage.
    :param blob_path: The path of the object in the Blob Storage. Must be something like "image/doggo.png".
    :param folder_path: The path to save the downloaded object.
    """
    # Create the file path for saving the downloaded image.
    file_path = os.path.join(folder_path, blob_path.split("/")[-1])

    if not os.path.exists(file_path):
        binary_obj = get_file_in_bytes_from_blob(blob_path)

        # Write the binary image data to the file using pickle.
        with open(file_path, "wb") as file:
            # pickle.dump(binary_obj, file)
            file.write(binary_obj)


def upload_blob(file_name: str, path_to_save: str, local_file_path: str):
    """
    Upload a blob to the blob storage.
    :param file_name: The name of the file to upload.
    :param path_to_save: The path to save the file to.
    :param local_file_path: The path to the local file to upload.
    :return: The response from the server.
    """
    url = (
        ApiObjectsContainer.http_client.base_url
        + ApiObjectsContainer.api_filter.Storage.post_uploadBlob.route
        + f"?fileName={file_name}&pathToSave={path_to_save}"
    )

    headers = ApiObjectsContainer.http_client.headers.copy()
    headers.pop("Content-Type", None)

    with open(local_file_path, "rb") as f:
        data = f.read()
        response = requests.post(url, headers=headers, verify=False, data=data)

    if response.status_code == 202 or response.status_code == 200:
        return response.json()
    else:
        raise Exception(
            f"Failed to upload blob: {response.status_code} {response.text}"
        )


def save_blob(blob_path: str, local_file_path: str):
    """
    Save a blob to the blob storage.
    :param blob_path: The path of the blob to save.
    :param local_file_path: The path to the local file to save.
    """
    file_name = os.path.basename(local_file_path)
    path_to_save = os.path.dirname(blob_path)
    return upload_blob(
        file_name=file_name, path_to_save=path_to_save, local_file_path=local_file_path
    )


def get_upload_progress(task_id: str):
    """
    Get the progress of an upload task.
    :param task_id: The ID of the upload task.
    :return: The progress of the upload task.
    """
    return ApiObjectsContainer.api_filter.Storage.get_getUploadProgress_with_taskId.run(
        ApiObjectsContainer.http_client, taskId=task_id
    )
