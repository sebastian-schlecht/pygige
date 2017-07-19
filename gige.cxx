#include <gige.hpp>

gige::IDevice setup(void) {
	// number of images to be allocated in buffer
	UINT32 numImages = 10;
	// allocate image buffer
	UINT8** buffer = new UINT8*[numImages];
	if (buffer != NULL) {
		for (UINT32 i = 0; i < numImages; ++i) {
			buffer[i] = NULL;
		}
	}


	gige::InitGigEVisionAPI();
	gige::IGigEVisionAPI gigeApi = gige::GetGigEVisionAPI();

	if (!gigeApi->IsUsingKernelDriver()) {
		std::cout << "Warning: Smartek Filter Driver not loaded." << std::endl;
	}

	gigeApi->SetHeartbeatTime(3);
	// gige_IGigEVisionAPI_RegisterCallback(gigeApi, gige_ICallbackEvent_Handler);

	// discover all devices on network
	gigeApi->FindAllDevices(3.0);
	const gige::DevicesList devices = gigeApi->GetAllDevices();

	if (devices.size() > 0 && buffer != NULL) {
		// take first device in list
		gige::IDevice device = devices[0];
		
		if (device != NULL && device->Connect()) {
			std::string text;
			INT64 int64Value;
			std::cout << "Connected to first camera: " << Common::IpAddrToString(device->GetIpAddress()) << std::endl;
			if (device->GetStringNodeValue("DeviceVendorName", text)) {
				std::cout << "Device Vendor: " << text << std::endl;
			}
			if (device->GetStringNodeValue("DeviceModelName", text)) {
				std::cout << "Device Model: " << text << std::endl;
			}
			if (device->GetIntegerNodeValue("Width", int64Value)) {
				std::cout << "Width: " << (int)int64Value << std::endl;
			}
			if (device->GetIntegerNodeValue("Height", int64Value)) {
				std::cout << "Height: " << (int)int64Value << std::endl;
			}

			INT64 packetSize = 0;
			device->GetIntegerNodeValue("GevSCPSPacketSize", packetSize);
			packetSize = packetSize & 0xFFFF;
			std::cout	<< "PacketSize: " << (int)packetSize << std::endl;

			// reset image queue to zero
			device->ResetImageQueue();
			// allocate user image queue and prepare for image buffer
			device->InitUserImageQueue(numImages);
			// get image size from camera
			INT64 payloadSize;
			device->GetIntegerNodeValue("PayloadSize", payloadSize);

			for (UINT32 i = 0; i < numImages; ++i) {
				// allocate memory for image
				buffer[i] = new UINT8[payloadSize];
				// add memory to image queue
				if (buffer[i] != NULL)
					device->AddToUserImageQueue(buffer[i], payloadSize);
			}
			
			// disable trigger mode
			bool status = device->SetStringNodeValue("TriggerMode", "Off");
			// set continuous acquisition mode
			status = device->SetStringNodeValue("AcquisitionMode", "Continuous");
			// start acquisition
			status = device->SetIntegerNodeValue("TLParamsLocked", 1);
			status = device->CommandNodeExecute("AcquisitionStart");

			std::cout << "Acquisition Start, press any key to exit loop..." << std::endl;
			
			//return device;
			return device;
		}
		else {
			std::cout << "Warning:Could not connect to device!" << std::endl;
			return NULL;
		}
	}
	else {
		std::cout << "Warning: No devices found!" << std::endl;
		return NULL;
	}

}

/*
bool setStringProperty(gige::IDevice device, std::string property, std::string value) {
    return device->SetStringNodeValue(property, value);
}

std::string getStringProperty(gige::IDevice device, std::string property) {
    return device->GetStringNodeValue(property);
}

bool setIntegerProperty(gige::IDevice device, std::string property, int value) {
    return device->SetIntegerNodeValue(property, value);
}

int getIntegerProperty(gige::IDevice device, std::string property) {
    return device->GetStringNodeValue(property, );
}
*/

Frame getFrame(gige::IDevice device, float timeout) {
	// wait for image for 3 seconds
	Frame frame;
	if (device->WaitForImage(timeout)) {
		// grab one image
		gige::IImageInfo imageInfo;
		device->GetImageInfo(&imageInfo);

		if (imageInfo != NULL) {
			UINT32 sizeX, sizeY;
			imageInfo->GetSize(sizeX, sizeY);
			
			int size = imageInfo->GetRawDataSize();
			UINT8* buffer = new UINT8[size];
			std::memcpy(buffer, imageInfo->GetRawData(), sizeof(UINT8) * size);
			frame.data = buffer;

			bool result = imageInfo->GetSize(frame.x, frame.y);
			if(!result) {
				std::cout << "Failed to acquire image size" << std::endl;
			}
				
		}
		// remove (pop) image from image buffer
		device->PopImage(imageInfo);
	}
	else {
		std::cout << "No image acquired..." << std::endl;
	}
	return frame;
}

void clearBuffer(UINT8* buffer) {
	delete [] buffer;
}


void close(gige::IDevice device) {
	device->Disconnect();
	std::cout << "Disconnected from " << Common::IpAddrToString(device->GetIpAddress()) << std::endl;	
}