# Tek komutla pipeline. Windows kullanicilari icin 'python run_all.py' onerilir
# (make gerektirmez). Linux/Mac'te asagidaki hedefler kullanilabilir.
PYTHON ?= python

.PHONY: core all test

core:        ## Cekirdek hat (scripts 01-10)
	$(PYTHON) run_all.py

all:         ## Cekirdek + ek dogrulama/saglamlik betikleri (11-14)
	$(PYTHON) run_all.py --all

test:        ## Birim testleri
	$(PYTHON) -m pytest -q
