# Worker capability matrix

| Profile | API | Core | Media | Multimodal | ffmpeg | Torch/CUDA |
|---|---:|---:|---:|---:|---:|---:|
| api | yes | no | no | no | no | no |
| core | no | yes | no | no | no | no |
| media | no | no | yes | no | yes | no |
| multimodal | no | no | no | yes | yes | yes |

The routing policy is immutable and lives in `domain.worker_capability`; an
unsupported task must fail closed rather than execute in another profile.
