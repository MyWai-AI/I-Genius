# FIWARE Integration

## 1. Purpose

This directory documents the FIWARE-specific integration included in this
repository.

The FIWARE module provides a northbound interface for representing trajectory
execution metadata and lifecycle status through the NGSI-LD API and an
Orion-LD context broker. It is intentionally separated from the runtime robot
communication path.

## 2. Architectural Role

The architecture in this branch is organized as follows:

- `VILMA`: user-facing learning-from-demonstration toolkit and orchestration
  layer
- `Vulcanexus`: southbound runtime transport layer for robot-side delivery
- `edge receiver`: robot-side intake and status feedback layer
- `FIWARE / Orion-LD`: northbound API and execution-status layer

Within this architecture:

- `Vulcanexus` transports the runtime trajectory payload
- `FIWARE / Orion-LD` stores and exposes execution metadata and status

The FIWARE module does not replace the runtime transport layer and does not
transfer raw trajectories to the robot in the current implementation.

## 3. Standards and APIs

This integration is based on the following standards and components:

- `NGSI-LD`: ETSI standard for context information management
- `Orion-LD`: FIWARE context broker implementing the NGSI-LD API

Reference sources:

- ETSI NGSI-LD specification:
  `https://cim.etsi.org/NGSI-LD/official/clause-4.html`
- FIWARE Orion-LD repository:
  `https://github.com/FIWARE/context.Orion-LD`

The main HTTP operations used in this repository are:

- `POST /ngsi-ld/v1/entities`
- `PATCH /ngsi-ld/v1/entities/{id}/attrs`
- `GET /ngsi-ld/v1/entities?type=TrajectoryExecution`

## 4. Repository Components

The FIWARE-related implementation is distributed across the following files:

- `docker-compose.fiware.yml`
- `src/streamlit_template/new_ui/pages/Common/fiware_page.py`
- `src/streamlit_template/new_ui/services/Common/fiware_service.py`
- `src/streamlit_template/new_ui/pages/Common/landing_page.py`

## 5. NGSI-LD Entity Model

The current implementation models an execution as a `TrajectoryExecution`
entity.

Typical attributes include:

- `executionId`
- `robotId`
- `skillId`
- `status`
- `transportProfile`
- `topicName`
- `statusTopic`
- `discoveryServer`
- `poseCount`
- `statusMessage`
- `updatedAt`

This entity provides an API-level representation of execution state without
coupling FIWARE to the southbound transport mechanism.

## 6. Local Deployment

The repository includes a dedicated local FIWARE stack:

- `docker-compose.fiware.yml`

The tested image versions are pinned as:

- `quay.io/fiware/orion-ld:1.10.0`
- `mongo:4.2`

These versions were selected to provide a stable local test environment and to
avoid incompatibilities observed with floating image tags.

### Start the broker stack

```bash
cd /home/vvijaykumar/vilma-agent
docker compose -f docker-compose.fiware.yml up -d
```

### Optional health check

```bash
curl -i http://127.0.0.1:1026/version
```

`Orion-LD` is an API service rather than a browser-based application. Useful
verification endpoints are therefore located under `/ngsi-ld/v1/...`.

## 7. UI Operation

To use the FIWARE module through the VILMA interface:

1. Start the VILMA UI.
2. Open the `FIWARE` page.
3. Keep `Orion-LD URL` set to `http://127.0.0.1:1026`.
4. Review or modify the execution payload fields.
5. Select `Upsert Execution` to create or update the entity.
6. Optionally update the status using `Patch Status Only`.

The FIWARE page is intentionally manual. This preserves separation between the
northbound API layer and the runtime execution path.

## 8. Verification

### List execution entities

```bash
curl -s 'http://127.0.0.1:1026/ngsi-ld/v1/entities?type=TrajectoryExecution' \
  -H 'Accept: application/ld+json'
```

### Retrieve a specific entity

```bash
curl -s 'http://127.0.0.1:1026/ngsi-ld/v1/entities/<entity-id>' \
  -H 'Accept: application/ld+json'
```

Example:

```bash
curl -s 'http://127.0.0.1:1026/ngsi-ld/v1/entities/urn:ngsi-ld:TrajectoryExecution:exec-20260409-140253' \
  -H 'Accept: application/ld+json'
```

### Expected result

After a successful execution upsert:

- at least one `TrajectoryExecution` entity is returned
- the entity includes fields such as:
  - `executionId`
  - `robotId`
  - `status`
  - `transportProfile`
  - `updatedAt`

After a successful status patch:

- the entity reflects the updated `status`
- `updatedAt` changes accordingly

## 9. Reusability

The FIWARE module is reusable for the following reasons:

- the broker endpoint is configurable
- the same UI and client logic can target any Orion-LD deployment
- the entity model is based on NGSI-LD rather than a project-specific API
- the integration remains independent of robot-specific execution logic

To adapt this module to another deployment, it is generally sufficient to
change:

- Orion-LD URL
- tenant, if applicable
- JSON-LD context, if applicable
- execution payload values

The core UI and HTTP client implementation can remain unchanged.

## 10. Scope and Current Limitations

The current implementation provides:

- a standalone FIWARE page in the VILMA UI
- Orion-LD entity create and update support
- execution status patch support
- a local broker stack for testing
- local verification through NGSI-LD HTTP calls

The current implementation does not provide:

- raw trajectory transport through FIWARE
- replacement of Vulcanexus as the runtime transport layer
- automatic broker updates for every runtime event
- the full FIWARE ecosystem such as historical data services

## 11. Summary

In this branch, the FIWARE integration should be understood as:

- `Vulcanexus`: reusable southbound runtime transport
- `FIWARE / Orion-LD`: reusable northbound API and status layer

This separation is intentional and supports a clearer layered architecture for
ARISE-aligned integration work.
