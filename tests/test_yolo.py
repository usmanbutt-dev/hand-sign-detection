"""
Tests for Phase 3: YOLO training utilities and data preparation.

Strategy: test everything EXCEPT the actual YOLO training call
(which requires model weights + GPU time and is not suitable for CI).
We mock the YOLO model where needed.
"""




# ─── YOLO Dataset Format Tests ───────────────────────────────────────────────

class TestYoloDataFormat:
    """Test that YOLO label generation is correct."""

    def test_yolo_label_format(self, tmp_path):
        """YOLO labels must be: class_id cx cy w h, all in [0,1]."""
        from src.data.prepare_yolo import generate_synthetic_yolo_data

        ds_dir = generate_synthetic_yolo_data(
            output_dir=tmp_path / "yolo_test",
            n_train=5, n_val=2, n_test=1, imgsz=128
        )

        lbl_dir = ds_dir / "labels" / "train"
        lbl_files = list(lbl_dir.glob("*.txt"))
        assert len(lbl_files) == 5, f"Expected 5 label files, got {len(lbl_files)}"

        for lbl_file in lbl_files:
            lines = lbl_file.read_text().strip().split("\n")
            for line in lines:
                parts = line.split()
                assert len(parts) == 5, f"YOLO label must have 5 values: {line}"
                class_id = int(parts[0])
                cx, cy, w, h = [float(x) for x in parts[1:]]

                assert class_id == 0, "Hand class_id must be 0"
                assert 0.0 <= cx <= 1.0, f"cx out of range: {cx}"
                assert 0.0 <= cy <= 1.0, f"cy out of range: {cy}"
                assert 0.0 < w <= 1.0,  f"w out of range: {w}"
                assert 0.0 < h <= 1.0,  f"h out of range: {h}"

    def test_image_and_label_counts_match(self, tmp_path):
        """Every image must have exactly one label file."""
        from src.data.prepare_yolo import generate_synthetic_yolo_data

        ds_dir = generate_synthetic_yolo_data(
            output_dir=tmp_path / "yolo_test",
            n_train=10, n_val=3, n_test=2, imgsz=64
        )

        for split in ["train", "val", "test"]:
            imgs = list((ds_dir / "images" / split).glob("*.jpg"))
            lbls = list((ds_dir / "labels" / split).glob("*.txt"))
            assert len(imgs) == len(lbls), (
                f"{split}: {len(imgs)} images but {len(lbls)} labels"
            )

    def test_directory_structure_correct(self, tmp_path):
        """YOLO expects images/train, images/val, labels/train, labels/val."""
        from src.data.prepare_yolo import generate_synthetic_yolo_data

        ds_dir = generate_synthetic_yolo_data(
            output_dir=tmp_path / "yolo_test",
            n_train=2, n_val=1, n_test=1, imgsz=64
        )

        expected_dirs = [
            ds_dir / "images" / "train",
            ds_dir / "images" / "val",
            ds_dir / "images" / "test",
            ds_dir / "labels" / "train",
            ds_dir / "labels" / "val",
            ds_dir / "labels" / "test",
        ]
        for d in expected_dirs:
            assert d.exists(), f"Missing required YOLO directory: {d}"


# ─── Coordinate Conversion Tests ─────────────────────────────────────────────

class TestCoordinateConversion:
    """Test COCO → YOLO coordinate conversion logic."""

    def test_center_pixel_bbox_converts_correctly(self):
        """A box at center of a 640×640 image should give cx=cy=0.5."""
        imgsz = 640
        # COCO format: x_min=160, y_min=160, w=320, h=320 (centered square)
        x_min, y_min, bw, bh = 160, 160, 320, 320

        cx = (x_min + bw / 2) / imgsz
        cy = (y_min + bh / 2) / imgsz
        nw = bw / imgsz
        nh = bh / imgsz

        assert abs(cx - 0.5) < 1e-6, f"cx should be 0.5, got {cx}"
        assert abs(cy - 0.5) < 1e-6, f"cy should be 0.5, got {cy}"
        assert abs(nw - 0.5) < 1e-6, f"nw should be 0.5, got {nw}"
        assert abs(nh - 0.5) < 1e-6, f"nh should be 0.5, got {nh}"

    def test_top_left_bbox_converts_correctly(self):
        """A box at top-left should give cx=cy < 0.5."""
        imgsz = 640
        x_min, y_min, bw, bh = 0, 0, 100, 100

        cx = (x_min + bw / 2) / imgsz
        cy = (y_min + bh / 2) / imgsz

        assert cx < 0.5
        assert cy < 0.5
        assert cx > 0.0  # Not exactly 0 since center is at 50px

    def test_all_values_in_unit_range(self):
        """All converted values must be in [0, 1]."""
        imgsz = 640
        test_cases = [
            (0, 0, 640, 640),      # Full image
            (0, 0, 1, 1),          # Tiny box at corner
            (320, 320, 100, 100),  # Center box
        ]
        for x_min, y_min, bw, bh in test_cases:
            cx = (x_min + bw / 2) / imgsz
            cy = (y_min + bh / 2) / imgsz
            nw = bw / imgsz
            nh = bh / imgsz
            for val, name in [(cx, "cx"), (cy, "cy"), (nw, "nw"), (nh, "nh")]:
                assert 0.0 <= val <= 1.0, f"{name}={val} out of [0,1] for bbox {x_min,y_min,bw,bh}"


# ─── Dataset YAML Tests ──────────────────────────────────────────────────────

class TestDatasetYaml:

    def test_yaml_created_with_correct_fields(self, tmp_path):
        """Dataset YAML must have path, train, val, nc, names."""
        import yaml

        from src.training.train_yolo import build_dataset_yaml

        yaml_path = build_dataset_yaml(
            data_dir=tmp_path,
            output_path=tmp_path / "dataset.yaml",
            class_names=["hand"],
        )

        assert yaml_path.exists()
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        assert "path" in cfg
        assert "train" in cfg
        assert "val" in cfg
        assert "nc" in cfg
        assert "names" in cfg
        assert cfg["nc"] == 1
        assert cfg["names"][0] == "hand"

    def test_yaml_nc_matches_class_count(self, tmp_path):
        """nc field must equal len(class_names)."""
        import yaml

        from src.training.train_yolo import build_dataset_yaml

        yaml_path = build_dataset_yaml(
            data_dir=tmp_path,
            output_path=tmp_path / "dataset.yaml",
            class_names=["hand", "face", "person"],
        )
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        assert cfg["nc"] == 3


# ─── MLflow Utils Tests ──────────────────────────────────────────────────────

class TestMlflowUtils:

    def test_log_run_returns_run_id(self, tmp_path, monkeypatch):
        """Logging a run should return a valid run ID string."""
        import mlflow

        from src.utils.mlflow_utils import log_yolo_training_run

        monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
        # Use plain path (not file:// URI) — MLflow 3.x rejects file:// on Windows
        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))

        run_id = log_yolo_training_run(
            run_name="test_run",
            hyperparams={"epochs": 5, "batch": 8, "model": "yolo11n"},
            metrics={"mAP50": 0.85, "precision": 0.90, "recall": 0.80},
            experiment_name="test-experiment",
        )

        assert isinstance(run_id, str)
        assert len(run_id) > 0

    def test_experiment_created_if_not_exists(self, tmp_path, monkeypatch):
        """Creating a new experiment should not raise any error."""
        import mlflow

        from src.utils.mlflow_utils import get_or_create_experiment

        monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        exp_id = get_or_create_experiment("brand-new-experiment-xyz")
        assert exp_id is not None

    def test_reusing_experiment_does_not_duplicate(self, tmp_path, monkeypatch):
        """Calling get_or_create_experiment twice should return same ID."""
        import mlflow

        from src.utils.mlflow_utils import get_or_create_experiment

        monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        exp_id_1 = get_or_create_experiment("my-experiment")
        exp_id_2 = get_or_create_experiment("my-experiment")
        assert exp_id_1 == exp_id_2
