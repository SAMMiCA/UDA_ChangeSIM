import os

data_path = '/media/smyoo/Backup_Data/dataset/Indoor/Mapping'

unnecessary_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_path)) for f in
                                            fn if (('/raw' in dp) or ('.db' in f))]
# print(unnecessary_files[0])

for file in unnecessary_files:
    os.remove(file)

# train_list = ['Warehouse_01/Room_0', 'Warehouse/Room_0', 'ModularWarehouse/Room_1', 'Storage/Room_0', 'StorageHouse/Room_1', 'StorageHouse/Room_2']
# test_list = ['Warehouse_01/Room_1', 'StorageHouse/Room_0', 'Storage/Room_1', 'ModularWarehouse/Room_0']
#
# image_total_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_path)) for f in
#                                     fn if (('rgb' in dp) and (any(map in dp for map in train_list)))]
#
# print(image_total_files)