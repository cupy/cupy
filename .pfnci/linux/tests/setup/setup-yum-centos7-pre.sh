#!/bin/bash -uex

yum-config-manager --add-repo http://ftp.jaist.ac.jp/pub/Linux/CentOS-vault/7.9.2009/os/x86_64/
yum-config-manager --add-repo http://ftp.jaist.ac.jp/pub/Linux/CentOS-vault/7.9.2009/updates/x86_64/
yum-config-manager --add-repo http://ftp.jaist.ac.jp/pub/Linux/CentOS-vault/7.9.2009/extras/x86_64/
yum-config-manager --add-repo http://ftp.jaist.ac.jp/pub/Linux/CentOS-vault/7.9.2009/centosplus/x86_64/
yum-config-manager --add-repo http://ftp.jaist.ac.jp/pub/Linux/CentOS-vault/7.9.2009/sclo/x86_64/rh

yum-config-manager --disable 'CentOS-7 - Base'
yum-config-manager --disable 'CentOS-7 - Extras'
yum-config-manager --disable 'CentOS-7 - Updates'

rpm --import https://www.centos.org/keys/RPM-GPG-KEY-CentOS-SIG-SCLo
