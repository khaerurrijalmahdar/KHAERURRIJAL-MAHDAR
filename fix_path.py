import sqlite3

conn = sqlite3.connect('data_pdam.db')
cursor = conn.cursor()

cursor.execute("UPDATE pencatatan_meteran SET gambar = REPLACE(gambar, 'uploads/', '')")
conn.commit()
conn.close()
print("Sukses hapus prefix 'uploads/' dari data lama.")
