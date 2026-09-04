from pathlib import Path
import zipfile
import h5py
import tempfile

model_path = Path(
    r"C:\Users\Kyle Eva\Downloads\datasci2-thesis\src\modeling\data\models\lstm_autoencoder_exp04.keras"
)

with zipfile.ZipFile(model_path, "r") as z:
    print("Files inside .keras:")
    print(z.namelist())

    with tempfile.TemporaryDirectory() as temp_dir:
        z.extract("model.weights.h5", temp_dir)

        weights_path = Path(temp_dir) / "model.weights.h5"

        print("\nWeight shapes:")
        with h5py.File(weights_path, "r") as f:
            def show(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"{name}: {obj.shape}")

            f.visititems(show)