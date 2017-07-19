#ifndef METHODS_HPP
#define METHODS_HPP

#pragma once

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <memory.h>
#include <cstring>

#include "Common.h"
#include <numpy/arrayobject.h>


struct Frame {
	UINT8* data;
	UINT32 x;
	UINT32 y;
};

gige::IDevice setup(void);
Frame getFrame(gige::IDevice device, float timeout);
void clearBuffer(UINT8* buffer);
void close(gige::IDevice device);


#endif