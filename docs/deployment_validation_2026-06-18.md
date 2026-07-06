# Deployment Validation - 2026-06-18

## Rollback Snapshot

Captured before recreating any container.

| Container | Current image reference | Current image ID | Status at audit | Created |
|---|---|---|---|---|
| `vilma-agent` | `mywaik8snextdev.azurecr.io/vilma:latest` | `sha256:8d03d3dae6ae1f2ff2a69499e02c05adbb3269a3589366edda808b8df452bdfa` | running | `2026-05-07T09:29:58.276598908Z` |
| `vulcanexus_humble` | `eprosima/vulcanexus:humble-desktop` | `sha256:dd5f0bce9b5ced55181391fedc21dc412699597adad55722806e21a1a24e3297` | running | `2026-05-29T12:09:42.759396839Z` |
| `zenoh_cloud` | `eclipse/zenoh-bridge-dds:latest` | `sha256:5e0b68304f99e02f7cfac78c11106e40f78728e26953e383b5f0d3743a70edfb` | running | `2026-06-11T16:52:47.4883977Z` |

Local image digests observed:

| Image | Tag | Repo digest | Image ID |
|---|---|---|---|
| `mywaik8snextdev.azurecr.io/vilma` | `latest` | `sha256:81acb6ed8bac37902f2c08df95600b701315743f820a91bbeac1abdeeae45598` | `8d03d3dae6ae` |
| `eprosima/vulcanexus` | `humble-desktop` | `sha256:c419e1e1013b9f2b7694ba20936bd0b49bc86d16ed6b6d6984e6ed06a7790599` | `dd5f0bce9b5c` |
| `eclipse/zenoh-bridge-dds` | `latest` | `sha256:390263ad3676dcf9a4fd9d45c6931ee2d839e7e070c62c13b38d1552575834ec` | `5e0b68304f99` |
| `vilma` | `latest` | none | `86bc103a9f6c` |

## Rollback Commands

If the recreated `vilma-agent` container fails, restore the previous app
container image without touching `vulcanexus_humble` or `zenoh_cloud`:

```bash
docker stop vilma-agent
docker rm vilma-agent
docker run -d \
  --name vilma-agent \
  --restart unless-stopped \
  -p 9002:8504 \
  mywaik8snextdev.azurecr.io/vilma:latest
```

If the previous deployment used additional environment variables or mounts,
inspect the saved old container config before removal when possible:

```bash
docker inspect vilma-agent
```

Do not recreate `vulcanexus_humble` or `zenoh_cloud` during app rollback.

## Validation Log

- Step 1 audit: completed.
- Step 2 Docker CLI image check: `docker run --rm vilma:latest which docker`
  returned `/usr/bin/docker`.
- Compatibility issue found before recreation: the existing production Compose
  file passes through `UV_INDEX_MYWAI_USERNAME`,
  `UV_INDEX_MYWAI_PASSWORD`, `MYWAI_USER`, `MYWAI_PASSWORD`, and
  `MYWAI_API_ENDPOINT`. The repo Compose service was updated to preserve these
  runtime environment variables before recreating `vilma-agent`.
