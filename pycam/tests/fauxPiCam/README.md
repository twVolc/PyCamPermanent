# Faux PiCam

## About

These files are used to generate a docker container running locally that can be used to test the direct file transfer and processing functions from a PiCam.
Specifically they produce a docker image this is: running Debian, has `proftpd` installed, has a pi user, has the requisite ports for a FTP connection open, and had a volume mounted on the host machine to add data to.

## Prerequsites

- [Install docker](https://www.docker.com/get-started/)
- Ensure docker and docker compose are installed and availble in the terminal:
  - `docker version`
  - `docker compose version`

## Usage

To build and run the container, in the terminal:

- Navigate to the direcotry contiang these files
- Run `docker compose up -d`
