FROM rockylinux:9.1 AS base

RUN dnf update -y && dnf install python3 -y
RUN dnf -y groupinstall development
RUN python3 -V

ENTRYPOINT ["sleep", "86400"]