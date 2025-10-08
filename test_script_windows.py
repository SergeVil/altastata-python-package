from altastata import AltaStataFunctions
import time

altastata_functions = AltaStataFunctions.from_account_dir('/Users/serge/.altastata/accounts/alice2.pqc')
altastata_functions.set_password("123");

# Store
result = altastata_functions.store(['/Users/serge/Desktop/desktop.ini', 
                                    '/Users/serge/Desktop/my_regedit.reg'], 
                                   'C:\\Users\\serge\\Desktop', 'StoreTest', True)

# Get the first CloudFileOperationStatus object
print('store: ' + str(result[0].getOperationStateValue()) + " " + result[0].getCloudFileCreateTime())

file_create_time_id = int(result[0].getCloudFileCreateTime())

users = ["bobtest1", "carolinetest3"]
result = altastata_functions.share_files('StoreTest/desktop.ini', True, None, None, users)
print('share_files:' + str(result[0].getOperationStateValue()))

# Retrieve file
result = altastata_functions.retrieve_files('/Users/serge/Desktop/tmp', 'StoreTest/desktop.ini', True, file_create_time_id, False, True)
print('retrieve_files:' + str(result[0].getOperationStateValue()))

# Show list
iterator = altastata_functions.list_cloud_files_versions('StoreTest', True, None, None)

for java_array in iterator:
    python_list = [str(element) for element in java_array]
    print(python_list)

# Read File as a buffer
buffer = altastata_functions.get_buffer('StoreTest/desktop.ini', file_create_time_id, 0, 4, 30)
print(buffer)

# Read File as a stream
input_stream = altastata_functions.get_java_input_stream('StoreTest/desktop.ini', file_create_time_id, 0, 100)

print("Input Stream: ")

# Read data
chunk_size = 32
while True:
    data = altastata_functions.get_buffer_from_input_stream(input_stream, chunk_size)
    if len(data) == 0:  # End of stream
        break
    # Process data
    print(data.decode('utf-8'), end="")

# Always explicitly call the close() method on the wrapper when you're done with it to prevent resource leaks
input_stream.close()

result = altastata_functions.get_file_attribute('StoreTest/desktop.ini', file_create_time_id, "readers")
print(f"readers: {result}")

result = altastata_functions.get_file_attribute('StoreTest/desktop.ini', file_create_time_id, "size")
print(f"size: {result}")


# Delete Files
result = altastata_functions.delete_files('StoreTest', True, None, None)
# Get the first CloudFileOperationStatus object
print('delete_files: ' + str(result[0].getOperationStateValue()))

altastata_functions.shutdown()

