# Data dictionary

This file documents implementation names used in the project-generated
manifests, tables, and receipts. The manuscript uses reader-facing scientific
terms instead.

## Injection-target fields

Fields beginning with `host_` refer to an injection target or to information
associated with that target:

- `host_galaxy_id`: injection-target identifier;
- `host_parent_id`: identifier of the target's parent scene or record;
- `host_spatial_group_id`: target spatial-group identifier;
- `host_physical_group_id`: target physical-system group identifier;
- `host_pixel_sha256`: checksum of the target pixel array; and
- `host_payload_sha256`: checksum of the target payload.

In manuscript prose, these objects are called **injection targets**, not
hosts. Statistical labels use **target spatial group** and **target-sampling
standard error**.

## Availability

This dictionary documents the local final products. It does not announce or
create a public release. The manuscript Data Availability section describes
the planned versioned release of redistributable project products by
publication. Third-party DR10 and COSMOS pixels remain subject to their
original archives and licences.
