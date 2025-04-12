CREATE DATABASE IF NOT EXISTS deportes_db;

USE deportes_db;

CREATE TABLE IF NOT EXISTS usuarios (
    usuario VARCHAR(255) PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL,
    clave VARCHAR(255) NOT NULL
);

-- Opcionalmente, insertar algunos usuarios iniciales
INSERT INTO usuarios (usuario, nombre, clave) VALUES
('admin', 'admin', 'admin'),
('MPAD', 'MPAD', 'MPAD'),
('luis', 'Luis', 'password123');
