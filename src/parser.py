import pickle
import jax.numpy as jnp
import grain.python as grain

def read_arrayrecord_file(file_path):
    # Use grain's ArrayRecordDataSource
    data_source = grain.ArrayRecordDataSource([file_path])
    
    print(f"Total records in file: {len(data_source)}")
    print("\n" + "="*50)
    
    # Read first record to inspect structure
    first_record = pickle.loads(data_source[0])
    
    print("First record structure:")
    for key, value in first_record.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)}")
    
    print("\n" + "="*50)
    
    # Example: Read a specific record
    sample_idx = 0
    sample_record = pickle.loads(data_source[sample_idx])
    
    print(f"\nTest {sample_idx}:")
    print(f"  img_indices shape: {sample_record['img_indices'].shape}")
    print(f"  img_indices sample: {sample_record['img_indices'][0][:10]}...")  # First 10 values
    print(f"  label shape: {sample_record['label'].shape}")
    print(f"  label sample: {sample_record['label'][:5]}")  # First 5 labels
    
    return data_source



if __name__ == "__main__":
    file_path = 'src/train_cifar10_indices.ar'
    
    # Inspect the file
    reader = read_arrayrecord_file(file_path)
    

    

    
