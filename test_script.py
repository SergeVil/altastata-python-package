from altastata import AltaStataFunctions
import time

altastata_functions = AltaStataFunctions.from_account_dir('/Users/sergevilvovsky/.altastata/accounts/amazon.rsa.alice222')
altastata_functions.set_password("123");

# Test create_file and append_buffer_to_file
print("\nTesting create_file and append_buffer_to_file:")

# Create an empty file
result = altastata_functions.create_file('StoreTest/empty_file.txt')
print('create empty file: ' + str(result.getOperationStateValue()) + " " + result.getCloudFileCreateTime())
file_create_time_id = int(result.getCloudFileCreateTime())

# Create a file with initial content
initial_content = b"Initial content\n"
result = altastata_functions.create_file('StoreTest/initial_content.txt', initial_content)
print('create file with content: ' + str(result.getOperationStateValue()) + " " + result.getCloudFileCreateTime())
initial_file_time_id = int(result.getCloudFileCreateTime())

# Append content to the empty file
append_content = b"Appended content\n"
altastata_functions.append_buffer_to_file('StoreTest/empty_file.txt', append_content, file_create_time_id)

# Verify the content by reading it back
buffer = altastata_functions.get_buffer('StoreTest/empty_file.txt', file_create_time_id, 0, 4, 100)
print("Appended file content: " + buffer.decode('utf-8'))

# Store
result = altastata_functions.store(['/Users/sergevilvovsky/Desktop/serge.png',
                                    '/Users/sergevilvovsky/Desktop/meeting_saved_chat.txt'],
                                   '/Users/sergevilvovsky/Desktop', 'StoreTest', True)

# Get the first CloudFileOperationStatus object
print('store: ' + str(result[0].getOperationStateValue()) + " " + result[0].getCloudFileCreateTime())

file_create_time_id = int(result[0].getCloudFileCreateTime())

users = ["bob123", "catrina777"]
result = altastata_functions.share_files('StoreTest/meeting_saved_chat', True, None, None, users)
print('share_files:' + str(result[0].getOperationStateValue()))

# Retrieve file
result = altastata_functions.retrieve_files('/Users/sergevilvovsky/Desktop/tmp', 'StoreTest/meeting_saved', True, file_create_time_id, False, True)
print('retrieve_files:' + str(result[0].getOperationStateValue()))

# Show list
iterator = altastata_functions.list_cloud_files_versions('StoreTest', True, None, None)

for java_array in iterator:
    python_list = [str(element) for element in java_array]
    print(python_list)

# Read File as a buffer
buffer = altastata_functions.get_buffer('StoreTest/meeting_saved_chat.txt', file_create_time_id, 0, 4, 100)
print("buffer: " + buffer.decode('utf-8'))

# Read File as a stream
input_stream = altastata_functions.get_java_input_stream('StoreTest/meeting_saved_chat.txt', file_create_time_id, 0, 100)

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

result = altastata_functions.get_file_attribute('StoreTest/meeting_saved_chat.txt', file_create_time_id, "access")
print(f"access: {result}")

result = altastata_functions.get_file_attribute('StoreTest/meeting_saved_chat.txt', file_create_time_id, "size")
print(f"size: {result}")

# Delete Files
result = altastata_functions.delete_files('StoreTest', True, None, None)
# Get the first CloudFileOperationStatus object
print('delete_files: ' + str(result[0].getOperationStateValue()))

altastata_functions.shutdown()

