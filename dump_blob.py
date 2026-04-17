import inspect
import sys
import mywai_python_integration_kit.apis.services.blob as b

source = inspect.getsource(sys.modules[b.__name__])
with open("blob_source.py", "w") as f:
    f.write(source)

# Let's also dump the storage endpoints
try:
    from mywai_python_integration_kit.apis.configuration import ApiObjectsContainer
    with open("storage_routes.txt", "w") as f:
        for attr in dir(ApiObjectsContainer.api_filter.Storage):
            if not attr.startswith('_'):
                val = getattr(ApiObjectsContainer.api_filter.Storage, attr)
                f.write(f"{attr}: {getattr(val, 'route', 'no route')}\n")
except Exception as e:
    with open("storage_routes.txt", "w") as f:
        f.write(f"Error: {e}")
