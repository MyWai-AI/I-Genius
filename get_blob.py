import inspect
import mywai_python_integration_kit.apis.services.blob as b
with open("kit_blob.txt", "w") as f:
    f.write(b.__file__ + "\n\n")
    if hasattr(b, "download_blob"):
        f.write(inspect.getsource(b.download_blob) + "\n\n")
    if hasattr(b, "save_blob"):
        f.write(inspect.getsource(b.save_blob) + "\n\n")
    if hasattr(b, "upload_blob"):
        f.write(inspect.getsource(b.upload_blob) + "\n\n")
