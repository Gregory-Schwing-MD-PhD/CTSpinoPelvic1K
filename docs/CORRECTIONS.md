## Ground-truth corrections to the source labels

The spinal-column labels in this dataset originate from **CTSpine1K** and the pelvic labels from **CTPelvic1K**. During quality control we identified and corrected a number of genuine errors in those original annotations (split vertebrae, skipped levels, and vertebrae left unlabelled within the field of view). Because the spine carries no pseudolabels, every spine correction below is a fix to the original CTSpine1K ground truth.

- **CTSpine1K (spine):** 112 corrections across 103 cases
- **CTPelvic1K (pelvis):** 13 corrections across 12 cases

| case | source | structure | error | detail | correction |
|---|---|---|---|---|---|
| 101 | CTPelvic1K | left_hip | split_into_pieces | labelled as 34% main body + a disconnected stray piece | pending (in review) |
| 121 | CTPelvic1K | right_hip | split_into_pieces | labelled as 36% main body + a disconnected stray piece | pending (in review) |
| 150 | CTPelvic1K | left_hip | split_into_pieces | labelled as 22% main body + a disconnected stray piece | pending (in review) |
| 248 | CTPelvic1K | S1 | split_into_pieces | labelled as 24% main body + a disconnected stray piece | pending (in review) |
| 267 | CTPelvic1K | left_hip | split_into_pieces | labelled as 49% main body + a disconnected stray piece | pending (in review) |
| 44 | CTPelvic1K | left_hip | split_into_pieces | labelled as 31% main body + a disconnected stray piece | pending (in review) |
| 525 | CTPelvic1K | left_hip | split_into_pieces | labelled as 17% main body + a disconnected stray piece | pending (in review) |
| 586 | CTPelvic1K | right_hip | split_into_pieces | labelled as 44% main body + a disconnected stray piece | pending (in review) |
| 723 | CTPelvic1K | left_hip | split_into_pieces | labelled as 34% main body + a disconnected stray piece | pending (in review) |
| 723 | CTPelvic1K | right_hip | split_into_pieces | labelled as 17% main body + a disconnected stray piece | pending (in review) |
| 778 | CTPelvic1K | right_hip | split_into_pieces | labelled as 23% main body + a disconnected stray piece | pending (in review) |
| 8 | CTPelvic1K | right_hip | split_into_pieces | labelled as 32% main body + a disconnected stray piece | pending (in review) |
| 9 | CTPelvic1K | left_hip | split_into_pieces | labelled as 20% main body + a disconnected stray piece | pending (in review) |
| 233 | CTSpine1K | T7 | missing_vertebra | a vertebra is missing between two labelled levels (skipped in the original annotation) | pending (in review) |
| 306 | CTSpine1K | T9 | missing_vertebra | a vertebra is missing between two labelled levels (skipped in the original annotation) | pending (in review) |
| 311 | CTSpine1K | T8 | missing_vertebra | a vertebra is missing between two labelled levels (skipped in the original annotation) | pending (in review) |
| 482 | CTSpine1K | T8 | missing_vertebra | a vertebra is missing between two labelled levels (skipped in the original annotation) | pending (in review) |
| 7 | CTSpine1K | T8 T9 | missing_vertebra | a vertebra is missing between two labelled levels (skipped in the original annotation) | pending (in review) |
| 746 | CTSpine1K | T7 | missing_vertebra | a vertebra is missing between two labelled levels (skipped in the original annotation) | pending (in review) |
| 103 | CTSpine1K | T11 | split_into_pieces | labelled as 18% main body + a disconnected stray piece | pending (in review) |
| 103 | CTSpine1K | L4 | split_into_pieces | labelled as 31% main body + a disconnected stray piece | pending (in review) |
| 112 | CTSpine1K | T8 | split_into_pieces | labelled as 40% main body + a disconnected stray piece | pending (in review) |
| 128 | CTSpine1K | T6 | split_into_pieces | labelled as 30% main body + a disconnected stray piece | pending (in review) |
| 136 | CTSpine1K | T8 | split_into_pieces | labelled as 46% main body + a disconnected stray piece | pending (in review) |
| 138 | CTSpine1K | T7 | split_into_pieces | labelled as 25% main body + a disconnected stray piece | pending (in review) |
| 14 | CTSpine1K | T8 | split_into_pieces | labelled as 34% main body + a disconnected stray piece | pending (in review) |
| 143 | CTSpine1K | T9 | split_into_pieces | labelled as 31% main body + a disconnected stray piece | pending (in review) |
| 144 | CTSpine1K | T8 | split_into_pieces | labelled as 39% main body + a disconnected stray piece | pending (in review) |
| 155 | CTSpine1K | L5 | split_into_pieces | labelled as 19% main body + a disconnected stray piece | pending (in review) |
| 169 | CTSpine1K | L1 | split_into_pieces | labelled as 23% main body + a disconnected stray piece | pending (in review) |
| 173 | CTSpine1K | L5 | split_into_pieces | labelled as 17% main body + a disconnected stray piece | pending (in review) |
| 179 | CTSpine1K | T9 | split_into_pieces | labelled as 40% main body + a disconnected stray piece | pending (in review) |
| 181 | CTSpine1K | T8 | split_into_pieces | labelled as 35% main body + a disconnected stray piece | pending (in review) |
| 220 | CTSpine1K | T9 | split_into_pieces | labelled as 28% main body + a disconnected stray piece | pending (in review) |
| 223 | CTSpine1K | T8 | split_into_pieces | labelled as 46% main body + a disconnected stray piece | pending (in review) |
| 23 | CTSpine1K | T8 | split_into_pieces | labelled as 43% main body + a disconnected stray piece | pending (in review) |
| 232 | CTSpine1K | T9 | split_into_pieces | labelled as 31% main body + a disconnected stray piece | pending (in review) |
| 234 | CTSpine1K | T9 | split_into_pieces | labelled as 44% main body + a disconnected stray piece | pending (in review) |
| 252 | CTSpine1K | T9 | split_into_pieces | labelled as 41% main body + a disconnected stray piece | pending (in review) |
| 255 | CTSpine1K | T6 | split_into_pieces | labelled as 19% main body + a disconnected stray piece | pending (in review) |
| 256 | CTSpine1K | T7 | split_into_pieces | labelled as 42% main body + a disconnected stray piece | pending (in review) |
| 261 | CTSpine1K | T9 | split_into_pieces | labelled as 33% main body + a disconnected stray piece | pending (in review) |
| 262 | CTSpine1K | T8 | split_into_pieces | labelled as 36% main body + a disconnected stray piece | pending (in review) |
| 267 | CTSpine1K | T10 | split_into_pieces | labelled as 49% main body + a disconnected stray piece | pending (in review) |
| 273 | CTSpine1K | T7 | split_into_pieces | labelled as 30% main body + a disconnected stray piece | pending (in review) |
| 276 | CTSpine1K | L3 | split_into_pieces | labelled as 27% main body + a disconnected stray piece | pending (in review) |
| 286 | CTSpine1K | T10 | split_into_pieces | labelled as 31% main body + a disconnected stray piece | pending (in review) |
| 288 | CTSpine1K | T9 | split_into_pieces | labelled as 35% main body + a disconnected stray piece | pending (in review) |
| 292 | CTSpine1K | T8 | split_into_pieces | labelled as 35% main body + a disconnected stray piece | pending (in review) |
| 295 | CTSpine1K | L2 | split_into_pieces | labelled as 22% main body + a disconnected stray piece | pending (in review) |
| 305 | CTSpine1K | T9 | split_into_pieces | labelled as 44% main body + a disconnected stray piece | pending (in review) |
| 307 | CTSpine1K | T8 | split_into_pieces | labelled as 41% main body + a disconnected stray piece | pending (in review) |
| 337 | CTSpine1K | T8 | split_into_pieces | labelled as 37% main body + a disconnected stray piece | pending (in review) |
| 339 | CTSpine1K | T9 | split_into_pieces | labelled as 50% main body + a disconnected stray piece | pending (in review) |
| 351 | CTSpine1K | T11 | split_into_pieces | labelled as 45% main body + a disconnected stray piece | pending (in review) |
| 359 | CTSpine1K | T7 | split_into_pieces | labelled as 28% main body + a disconnected stray piece | pending (in review) |
| 369 | CTSpine1K | T9 | split_into_pieces | labelled as 37% main body + a disconnected stray piece | pending (in review) |
| 370 | CTSpine1K | T9 | split_into_pieces | labelled as 42% main body + a disconnected stray piece | pending (in review) |
| 374 | CTSpine1K | T8 | split_into_pieces | labelled as 19% main body + a disconnected stray piece | pending (in review) |
| 382 | CTSpine1K | T10 | split_into_pieces | labelled as 16% main body + a disconnected stray piece | pending (in review) |
| 397 | CTSpine1K | T9 | split_into_pieces | labelled as 16% main body + a disconnected stray piece | pending (in review) |
| 398 | CTSpine1K | T10 | split_into_pieces | labelled as 37% main body + a disconnected stray piece | pending (in review) |
| 435 | CTSpine1K | T10 | split_into_pieces | labelled as 23% main body + a disconnected stray piece | pending (in review) |
| 450 | CTSpine1K | T9 | split_into_pieces | labelled as 40% main body + a disconnected stray piece | pending (in review) |
| 459 | CTSpine1K | T7 | split_into_pieces | labelled as 41% main body + a disconnected stray piece | pending (in review) |
| 46 | CTSpine1K | L3 | split_into_pieces | labelled as 20% main body + a disconnected stray piece | pending (in review) |
| 478 | CTSpine1K | T8 | split_into_pieces | labelled as 40% main body + a disconnected stray piece | pending (in review) |
| 486 | CTSpine1K | T8 | split_into_pieces | labelled as 41% main body + a disconnected stray piece | pending (in review) |
| 499 | CTSpine1K | T8 | split_into_pieces | labelled as 31% main body + a disconnected stray piece | pending (in review) |
| 510 | CTSpine1K | T7 | split_into_pieces | labelled as 25% main body + a disconnected stray piece | pending (in review) |
| 511 | CTSpine1K | T9 | split_into_pieces | labelled as 41% main body + a disconnected stray piece | pending (in review) |
| 542 | CTSpine1K | T7 | split_into_pieces | labelled as 34% main body + a disconnected stray piece | pending (in review) |
| 561 | CTSpine1K | T10 | split_into_pieces | labelled as 16% main body + a disconnected stray piece | pending (in review) |
| 584 | CTSpine1K | T7 | split_into_pieces | labelled as 17% main body + a disconnected stray piece | pending (in review) |
| 592 | CTSpine1K | T11 | split_into_pieces | labelled as 39% main body + a disconnected stray piece | pending (in review) |
| 595 | CTSpine1K | T8 | split_into_pieces | labelled as 30% main body + a disconnected stray piece | pending (in review) |
| 614 | CTSpine1K | T8 | split_into_pieces | labelled as 33% main body + a disconnected stray piece | pending (in review) |
| 619 | CTSpine1K | T10 | split_into_pieces | labelled as 29% main body + a disconnected stray piece | pending (in review) |
| 621 | CTSpine1K | T7 | split_into_pieces | labelled as 48% main body + a disconnected stray piece | pending (in review) |
| 630 | CTSpine1K | T10 | split_into_pieces | labelled as 29% main body + a disconnected stray piece | pending (in review) |
| 631 | CTSpine1K | T8 | split_into_pieces | labelled as 20% main body + a disconnected stray piece | pending (in review) |
| 636 | CTSpine1K | T8 | split_into_pieces | labelled as 23% main body + a disconnected stray piece | pending (in review) |
| 639 | CTSpine1K | T9 | split_into_pieces | labelled as 50% main body + a disconnected stray piece | pending (in review) |
| 648 | CTSpine1K | T10 | split_into_pieces | labelled as 49% main body + a disconnected stray piece | pending (in review) |
| 65 | CTSpine1K | T8 | split_into_pieces | labelled as 27% main body + a disconnected stray piece | pending (in review) |
| 665 | CTSpine1K | T10 | split_into_pieces | labelled as 45% main body + a disconnected stray piece | pending (in review) |
| 670 | CTSpine1K | T7 | split_into_pieces | labelled as 50% main body + a disconnected stray piece | pending (in review) |
| 681 | CTSpine1K | T9 | split_into_pieces | labelled as 20% main body + a disconnected stray piece | pending (in review) |
| 687 | CTSpine1K | T6 | split_into_pieces | labelled as 34% main body + a disconnected stray piece | pending (in review) |
| 696 | CTSpine1K | T8 | split_into_pieces | labelled as 50% main body + a disconnected stray piece | pending (in review) |
| 702 | CTSpine1K | T10 | split_into_pieces | labelled as 45% main body + a disconnected stray piece | pending (in review) |
| 704 | CTSpine1K | T11 | split_into_pieces | labelled as 24% main body + a disconnected stray piece | pending (in review) |
| 731 | CTSpine1K | T8 | split_into_pieces | labelled as 44% main body + a disconnected stray piece | pending (in review) |
| 739 | CTSpine1K | T9 | split_into_pieces | labelled as 50% main body + a disconnected stray piece | pending (in review) |
| 742 | CTSpine1K | T7 | split_into_pieces | labelled as 48% main body + a disconnected stray piece | pending (in review) |
| 746 | CTSpine1K | T9 | split_into_pieces | labelled as 38% main body + a disconnected stray piece | pending (in review) |
| 747 | CTSpine1K | T8 | split_into_pieces | labelled as 48% main body + a disconnected stray piece | pending (in review) |
| 750 | CTSpine1K | T7 | split_into_pieces | labelled as 46% main body + a disconnected stray piece | pending (in review) |
| 753 | CTSpine1K | T8 | split_into_pieces | labelled as 29% main body + a disconnected stray piece | pending (in review) |
| 756 | CTSpine1K | T7 | split_into_pieces | labelled as 26% main body + a disconnected stray piece | pending (in review) |
| 758 | CTSpine1K | T7 | split_into_pieces | labelled as 46% main body + a disconnected stray piece | pending (in review) |
| 770 | CTSpine1K | T9 | split_into_pieces | labelled as 48% main body + a disconnected stray piece | pending (in review) |
| 770 | CTSpine1K | L5 | split_into_pieces | labelled as 18% main body + a disconnected stray piece | pending (in review) |
| 772 | CTSpine1K | T8 | split_into_pieces | labelled as 39% main body + a disconnected stray piece | pending (in review) |
| 778 | CTSpine1K | T7 | split_into_pieces | labelled as 29% main body + a disconnected stray piece | pending (in review) |
| 781 | CTSpine1K | T11 | split_into_pieces | labelled as 33% main body + a disconnected stray piece | pending (in review) |
| 84 | CTSpine1K | T7 | split_into_pieces | labelled as 28% main body + a disconnected stray piece | pending (in review) |
| CTC-1018399231 | CTSpine1K | T6 | split_into_pieces | labelled as 30% main body + a disconnected stray piece | pending (in review) |
| CTC-1018399231 | CTSpine1K | L5 | split_into_pieces | labelled as 18% main body + a disconnected stray piece | pending (in review) |
| CTC-3105782108 | CTSpine1K | T10 | split_into_pieces | labelled as 48% main body + a disconnected stray piece | pending (in review) |
| 140 | CTSpine1K | upper thoracic | thoracic_unlabelled_in_fov | thoracic vertebrae are in the field of view but were not annotated in the original labels | pending (in review) |
| 169 | CTSpine1K | upper thoracic | thoracic_unlabelled_in_fov | thoracic vertebrae are in the field of view but were not annotated in the original labels | pending (in review) |
| 173 | CTSpine1K | upper thoracic | thoracic_unlabelled_in_fov | thoracic vertebrae are in the field of view but were not annotated in the original labels | pending (in review) |
| 22 | CTSpine1K | upper thoracic | thoracic_unlabelled_in_fov | thoracic vertebrae are in the field of view but were not annotated in the original labels | pending (in review) |
| 254 | CTSpine1K | upper thoracic | thoracic_unlabelled_in_fov | thoracic vertebrae are in the field of view but were not annotated in the original labels | pending (in review) |
| 272 | CTSpine1K | upper thoracic | thoracic_unlabelled_in_fov | thoracic vertebrae are in the field of view but were not annotated in the original labels | pending (in review) |
| 276 | CTSpine1K | upper thoracic | thoracic_unlabelled_in_fov | thoracic vertebrae are in the field of view but were not annotated in the original labels | pending (in review) |
| 295 | CTSpine1K | upper thoracic | thoracic_unlabelled_in_fov | thoracic vertebrae are in the field of view but were not annotated in the original labels | pending (in review) |
| 302 | CTSpine1K | upper thoracic | thoracic_unlabelled_in_fov | thoracic vertebrae are in the field of view but were not annotated in the original labels | pending (in review) |
| 46 | CTSpine1K | upper thoracic | thoracic_unlabelled_in_fov | thoracic vertebrae are in the field of view but were not annotated in the original labels | pending (in review) |
| 512 | CTSpine1K | upper thoracic | thoracic_unlabelled_in_fov | thoracic vertebrae are in the field of view but were not annotated in the original labels | pending (in review) |
| 61 | CTSpine1K | upper thoracic | thoracic_unlabelled_in_fov | thoracic vertebrae are in the field of view but were not annotated in the original labels | pending (in review) |
| 618 | CTSpine1K | upper thoracic | thoracic_unlabelled_in_fov | thoracic vertebrae are in the field of view but were not annotated in the original labels | pending (in review) |
| 703 | CTSpine1K | upper thoracic | thoracic_unlabelled_in_fov | thoracic vertebrae are in the field of view but were not annotated in the original labels | pending (in review) |
| 725 | CTSpine1K | upper thoracic | thoracic_unlabelled_in_fov | thoracic vertebrae are in the field of view but were not annotated in the original labels | pending (in review) |
