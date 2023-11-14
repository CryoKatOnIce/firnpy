# -*- coding: utf-8 -*-
"""
Initialization file for firnpy package.
Re-coded after a previous file lost.

Created by: Kat Sejan 9th December 2022.
"""
#loading all firnpy modules when firnpy is imported
import firnpy.load, firnpy.analise, firnpy.plot

#loading all firnpy modules when firnpy is imported with *
_all_ = ['load', 'analise', 'plot']