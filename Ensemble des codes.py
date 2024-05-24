
import numpy as np
import cv2
import os
import glob
import sqlite3

image_path = r"
image_list = sorted(glob.glob(os.path.join(image_path, '*.jpg')))
np.set_printoptions(suppress=True, precision=2)


for step in range(1, 16):
    database_directory = image_path
    database_path = os.path.join(database_directory, f"glanum_pas_{step}.db")

    con = sqlite3.connect(database_path)
    cur = con.cursor()

    cur.execute(f'''CREATE TABLE IF NOT EXISTS GL1_20220210_pas_{step} (
                    image1_name TEXT,
                    image2_name TEXT,
                    dis_mag REAL
                )''')
    con.commit()
    con.close()

    displacements = []
    successful_pairs = 0  

    for ii in range(len(image_list) - step):
        image_name_1 = os.path.basename(image_list[ii])
        image_name_2 = os.path.basename(image_list[ii + step])
        print(f"Comparing images with step {step}: {image_name_1}, {image_name_2}")

        img1 = cv2.imread(os.path.join(image_path, image_name_1), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.resize(img1, None, fx=0.2, fy=0.2)
        img2 = cv2.imread(os.path.join(image_path, image_name_2), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.resize(img2, None, fx=0.2, fy=0.2)

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:500]

        if len(matches) >= 100:
            img_points1 = np.array([kp1[match.queryIdx].pt for match in matches], dtype=float)
            img_points2 = np.array([kp2[match.trainIdx].pt for match in matches], dtype=float)
            dis_vec = np.median(abs(img_points2 - img_points1), axis=0)
            dis_mag = np.linalg.norm(dis_vec)
            print("Displacement Magnitude:", dis_mag)

            displacements.append(dis_mag)
            con = sqlite3.connect(database_path)
            cur = con.cursor()
            cur.execute(f"INSERT INTO GL1_20220210_pas_{step} VALUES (?, ?, ?)", (image_name_1, image_name_2, dis_mag))
            con.commit()
            con.close()
            successful_pairs += 1

    if displacements:
        average_disparity = np.mean(displacements)
        total_pairs = len(image_list) - step
        percentage_successful = (successful_pairs / total_pairs) * 100
        print(f"Average Disparity Magnitude for step {step}: {average_disparity:.2f}")
        print(f"Percentage of pairs with calculated displacement for step {step}: {percentage_successful:.2f}%")
    else:
        print("No displacements calculated for step {step}.")

cv2.destroyAllWindows()







import sqlite3
import os

db_path = r"C:

con = sqlite3.connect(db_path)
cur = con.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cur.fetchall()

for table_name in tables:
    print(f"Table: {table_name[0]}")

    cur.execute(f"PRAGMA table_info({table_name[0]})")
    columns = cur.fetchall()
    column_names = [column[1] for column in columns]  
    print("Columns:", column_names)

    cur.execute(f"SELECT * FROM {table_name[0]} LIMIT 3")  
    rows = cur.fetchall()
    for row in rows:
        print(row)
    print("\n")  

cur.close()
con.close()






import sqlite3
import os

directory = r"C:

merged_db_path = os.path.join(directory, 'merged_database.db')

merged_con = sqlite3.connect(merged_db_path)
merged_cur = merged_con.cursor()

merged_cur.execute('''CREATE TABLE IF NOT EXISTS GL1_20220210_pas (
                        image1_name TEXT,
                        image2_name TEXT,
                        dis_mag REAL
                    )''')
merged_con.commit()

for step in range(1, 21):
    individual_db_path = os.path.join(directory, f"glanum_pas_{step}.db")
    if os.path.exists(individual_db_path):
        print(f"Processing: {individual_db_path}")  # 
        try:
            merged_cur.execute(f"ATTACH DATABASE '{individual_db_path}' AS source")
            print("Database attached.")  # 

            source_table_name = f"GL1_20220210_pas_{step}"
            merged_cur.execute(f"INSERT INTO GL1_20220210_pas SELECT * FROM source.{source_table_name}")
            merged_con.commit()
            print(f"Data copied from {source_table_name}.")  # 

        except sqlite3.DatabaseError as e:
            print(f"An error occurred: {e}")
        finally:
            merged_cur.execute("DETACH DATABASE source")
            print("Database detached.")  # 

merged_con.close()
print("Merge complete and database closed.")  # 





import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import os
import glob

directory_path = r"C
image_files = glob.glob(
    os.path.join(directory_path, '*.JPG'))  # 

base_model = ResNet50(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output))

all_features = []

for img_path in image_files:
    print(f"Processing image: {os.path.basename(img_path)}")  #

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array)
    features_vector = np.ravel(features)  
    all_features.append(features_vector) 

all_features_array = np.array(all_features)

output_path = os.path.join(directory_path, 'all_features.npy')
np.save(output_path, all_features_array)

print(f"All features have been extracted and saved to {output_path}")





import sqlite3
import os
import numpy as np
import glob

features_path = r"C:
db_path = r"C:

features = np.load(features_path)

image_directory = r"C:\Users\Thomas\PycharmProjects\pythonProject3\20220420_glanum_mmn"
image_files = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(image_directory, '*.jpg')))]

feature_dict = {name: features[i] for i, name in enumerate(image_files)}

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS GL1_20220210_features (
    features1 BLOB,
    features2 BLOB,
    dis_mag REAL
)
''')

cursor.execute('SELECT image1_name, image2_name, dis_mag FROM GL1_20220210_pas')
rows = cursor.fetchall()

new_data = []
for image1_name, image2_name, dis_mag in rows:
    features1 = feature_dict.get(image1_name, None)
    features2 = feature_dict.get(image2_name, None)
    if features1 is not None and features2 is not None:  # Check if both features are found
        new_data.append((features1.tobytes(), features2.tobytes(), dis_mag))

if new_data:
    cursor.executemany('INSERT INTO GL1_20220210_features (features1, features2, dis_mag) VALUES (?, ?, ?)', new_data)
    conn.commit()

conn.close()
print("Feature vectors have been successfully updated in the database.")




import sqlite3
import numpy as np
import os
from tqdm import tqdm  # Import tqdm for the progress bar

# Define paths
db_path = r"C
npy_output_path = r"C

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('SELECT features1, features2, dis_mag FROM GL1_20220210_features')
rows = cursor.fetchall()
conn.close()  

dtypes = np.dtype([
    ('features1', np.float32, (2048,)),  
    ('features2', np.float32, (2048,)),
    ('dis_mag', np.float32)
])

data = np.zeros(len(rows), dtype=dtypes)
for i, (features1_blob, features2_blob, dis_mag) in enumerate(tqdm(rows, desc="Processing records")):
    features1 = np.frombuffer(features1_blob, dtype=np.float32)
    features2 = np.frombuffer(features2_blob, dtype=np.float32)
    data[i] = (features1, features2, dis_mag)

np.save(npy_output_path, data)

print(f"Structured data has been successfully saved to {npy_output_path}.")
print("Shape of the matrix:", data.shape)




afficher numpy final

import numpy as np

npy_file_path = r"

matrix = np.load(npy_file_path, allow_pickle=True)  # 

if matrix.ndim != 2:
    print("The loaded data is not a 2D matrix. Attempting to reshape or process...")
    try:
        matrix = np.vstack(matrix)
    except Exception as e:
        print("Error reshaping the data into a 2D matrix:", e)
else:
    print("Loaded data is a 2D matrix.")

print("Displaying the matrix:")
print(matrix)

print("Shape of the matrix:", matrix.shape)





import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Concatenate

input_shape = (2048,)
input1 = Input(shape=input_shape)
input2 = Input(shape=input_shape)
concatenated = Concatenate(axis=-1)([input1, input2])
dense1 = Dense(128)(concatenated)
dense2 = Dense(64)(dense1)
dense3 = Dense(32)(dense2)
dense4 = Dense(16)(dense3)
dense5 = Dense(8)(dense4)
output = Dense(1)(dense5)
model = Model(inputs=[input1, input2], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
data = np.load(r"C:\Users\Thomas\PycharmProjects\pythonProject3\dossier_pas\structured_features_data.npy", allow_pickle=True)
features1, features2, labels = np.stack(data['features1']), np.stack(data['features2']), data['dis_mag']
train_size = int(0.8 * len(labels))
input_data_train, labels_train = [features1[:train_size], features2[:train_size]], labels[:train_size]
input_data_test, labels_test = [features1[train_size:], features2[train_size:]], labels[train_size:]
model.fit(input_data_train, labels_train, epochs=10, batch_size=32, validation_data=(input_data_test, labels_test))
model.save(r"C:\Users\Thomas\PycharmProjects\pythonProject3\trained_model_5", save_format='tf')







import numpy as np
import tensorflow as tf

model_path = r"
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

data_path = r"C:\Users\Thomas\PycharmProjects\pythonProject2\all_features.npy"
features = np.load(data_path)

print("Shape of loaded feature data:", features.shape)

if features.shape[0] < 2:
    print("Not enough feature vectors for comparison.")
else:
    features1 = features[0:1]  
    features2 = features[1:2]  

    prediction = model.predict([features1, features2])
    print(f"Predicted disparity magnitude between the first and second : {prediction.flatten()[0]}")