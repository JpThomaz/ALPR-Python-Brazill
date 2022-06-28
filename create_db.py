import sqlite3 as sql

con = sql.connect('banco_alpr.db')
cur = con.cursor()

cur.execute("DROP TABLE IF EXISTS historico_veiculos")

sql ='''CREATE TABLE "historico_veiculos" (
    "ID"	INTEGER PRIMARY KEY AUTOINCREMENT,
	"PLACA"	TEXT NOT NULL,
	"MODELO"	TEXT NOT NULL,
    "HORA_ENTRADA"	DATA TIME NOT NULL,
    "HORA_SAIDA"	DATA TIME NULL
)'''
cur.execute(sql)

#commit changes
con.commit()

#close the connection
con.close()