import socket
import struct

client_ip = "192.168.65.97"
port = 26001
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def poslji(x, y, z):
	""" Poslje koordinate zogice na simulink """
	vals = (x, y, z)
	packer = struct.Struct('f f f')
	bin_data = packer.pack(*vals)
	sock.sendto(bin_data, (client_ip, port))