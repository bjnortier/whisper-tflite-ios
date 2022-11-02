//
//  whisper-bridge.hpp
//  whisper-tflite
//
//  Created by Ben Nortier on 2022/11/02.
//

#ifndef whisper_bridge_hpp
#define whisper_bridge_hpp

#ifdef __cplusplus
extern "C" {
#endif

int process(const char* resource_path, const char* pcm_filename, float* result);
const char * whisper_token_to_str2(int token);

#ifdef __cplusplus
}
#endif

#endif
