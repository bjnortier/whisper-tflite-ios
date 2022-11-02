//
//  whisper-bridge.cpp
//  whisper-tflite
//
//  Created by Ben Nortier on 2022/11/02.
//

#include <iostream>
#include <fstream>
#include <thread>
#include <sys/time.h>

#include "whisper_bridge.hpp"
#include "whisper.hpp"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

int print_time_taken(const char* label, timeval start, timeval end) {
  long start_milliseconds = start.tv_sec * 1000LL + start.tv_usec / 1000;
  long end_milliseconds = end.tv_sec * 1000LL + end.tv_usec / 1000;
  printf("%s: %ld [ms] \n", label, end_milliseconds - start_milliseconds);
  return 0;
}

const char * whisper_token_to_str2(int token) {
    return whisper_token_to_str(token);
}

int process(const char* resource_path, const char* pcmfilename, float* result) {
    // set up file containing filters and vocab
    // see https://github.com/usefulsensors/openai-whisper/blob/main/tflite_minimal/minimal.cc
    std::string fname = resource_path + std::string("filters_vocab_gen.bin");
    auto fin = std::ifstream(fname, std::ios::binary);
    {
        uint32_t magic = 0;
        fin.read((char*)&magic, sizeof(magic));
        if (magic != 0x5553454e) {
            printf("%s: invalid vocab file '%s' (bad magic)\n", __func__,
            fname.c_str());
            return 0;
        }
    }

    // load mel filters
    whisper_filters filters;
    {
      fin.read((char*)&filters.n_mel, sizeof(filters.n_mel));
      fin.read((char*)&filters.n_fft, sizeof(filters.n_fft));

      filters.data.resize(filters.n_mel * filters.n_fft);
      fin.read((char*)filters.data.data(), filters.data.size() * sizeof(float));
    }

    // load vocab
    int32_t n_vocab = 0;
    std::string word;
    {
        fin.read((char*)&n_vocab, sizeof(n_vocab));
        g_vocab.n_vocab = n_vocab;
        printf("\nn_vocab:%d\n", (int)n_vocab);

        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            fin.read((char*)&len, sizeof(len));

            word.resize(len);
            fin.read((char*)word.data(), len);
            g_vocab.id_to_token[i] = word;
            // printf("len:%d",(int)len);
            // printf("'%s'\n", g_vocab.id_to_token[i].c_str());
        }

        g_vocab.n_vocab = 51864;  // add additional vocab ids
        if (g_vocab.is_multilingual()) {
            g_vocab.token_eot++;
            g_vocab.token_sot++;
            g_vocab.token_prev++;
            g_vocab.token_solm++;
            g_vocab.token_not++;
            g_vocab.token_beg++;
        }
        for (int i = n_vocab; i < g_vocab.n_vocab; i++) {
            if (i > g_vocab.token_beg) {
                word = "[_TT_" + std::to_string(i - g_vocab.token_beg) + "]";
            } else if (i == g_vocab.token_eot) {
                word = "[_EOT_]";
            } else if (i == g_vocab.token_sot) {
                word = "[_SOT_]";
            } else if (i == g_vocab.token_prev) {
                word = "[_PREV_]";
            } else if (i == g_vocab.token_not) {
                word = "[_NOT_]";
            } else if (i == g_vocab.token_beg) {
                word = "[_BEG_]";
            } else {
                word = "[_extra_token_" + std::to_string(i) + "]";
            }
            g_vocab.id_to_token[i] = word;
          // printf("%s: g_vocab[%d] = '%s'\n", __func__, i, word.c_str());
        }
    }

    // read audio & create mel
    whisper_mel mel;
    {
        std::string pcm_path = resource_path + std::string(pcmfilename);
        std::vector<float> pcmf32;
        {
          drwav wav;
          if (!drwav_init_file(&wav, pcm_path.c_str(), NULL)) {
            fprintf(stderr, "failed to open WAV file '%s' - check your input\n",
                    pcm_path.c_str());
            //  whisper_print_usage(argc, argv, {});
            return 3;
          }

          if (wav.channels != 1 && wav.channels != 2) {
            fprintf(stderr, "WAV file '%s' must be mono or stereo\n",
                    pcmfilename);
            return 4;
          }

          if (wav.sampleRate != WHISPER_SAMPLE_RATE) {
            fprintf(stderr, "WAV file '%s' must be 16 kHz\n", 
                    pcmfilename);
            return 5;
          }

          if (wav.bitsPerSample != 16) {
            fprintf(stderr, "WAV file '%s' must be 16-bit\n",
                    pcmfilename);
            return 6;
          }

          unsigned long long n = wav.totalPCMFrameCount;

          std::vector<int16_t> pcm16;
          pcm16.resize(n * wav.channels);
          drwav_read_pcm_frames_s16(&wav, n, pcm16.data());
          drwav_uninit(&wav);
          // convert to mono, float
          pcmf32.resize(n);
          if (wav.channels == 1) {
            for (int i = 0; i < n; i++) {
              pcmf32[i] = float(pcm16[i]) / 32768.0f;
            }
          } else {
            for (int i = 0; i < n; i++) {
              pcmf32[i] = float(pcm16[2 * i] + pcm16[2 * i + 1]) / 65536.0f;
            }
          }
        }

        // Hack if the audio file size is less than 30ms append with 0's
        pcmf32.resize((WHISPER_SAMPLE_RATE * WHISPER_CHUNK_SIZE), 0);
        if (!log_mel_spectrogram(pcmf32.data(), pcmf32.size(), WHISPER_SAMPLE_RATE,
                                 WHISPER_N_FFT, WHISPER_HOP_LENGTH, WHISPER_N_MEL,
                                 1, filters, mel)) {
          fprintf(stderr, "%s: failed to compute mel spectrogram\n", __func__);
          return -1;
        }

        printf("\nmel.n_len%ld\n", mel.n_len);
        printf("\nmel.n_mel:%d\n", mel.n_mel);
        for (int i = 0; i < 10; ++i) {
            printf("%d: %.3f\n", i, mel.data[i]);
        }
    }  // end of audio file processing
    memcpy(result, mel.data.data(), 240000 * 4);

    return 0;
}
