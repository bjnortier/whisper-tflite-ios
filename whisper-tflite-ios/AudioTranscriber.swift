//
//  AudioTranscriber.swift
//  whisper-tflite-ios
//
//  Created by Ben Nortier on 2022/11/02.
//

import Foundation
import TensorFlowLite

private func dataToInt32Array(_ data: Data) -> [Int32]? {
    guard data.count % MemoryLayout<Int32>.stride == 0 else { return nil }

    return data.withUnsafeBytes { .init($0.bindMemory(to: Int32.self)) }
}

func transcribe() {
    print("init")

    var interpreter: Interpreter!

    let result = UnsafeMutablePointer<Float32>.allocate(capacity: 240000)
    defer {
        result.deallocate()
    }

    let resourcePath = Bundle.main.resourcePath! + "/"
    let wavFilename = "jfk.wav"
    process(resourcePath, wavFilename, result)

    guard let modelPath = Bundle.main.path(
        forResource: "whisper",
        ofType: "tflite"
    ) else { return }

    do {
        interpreter = try Interpreter(modelPath: modelPath)
        try interpreter.allocateTensors()
        let inputShape = try interpreter.input(at: 0).shape
        print("Input Shape: \(inputShape)")

        let floatData = Array(UnsafeBufferPointer(start: result, count: 240000))

        let audioBufferData = floatData.withUnsafeBufferPointer(Data.init)
        print("Buffer size: \(audioBufferData.count)")

        try interpreter.copy(audioBufferData, toInputAt: 0)
        try interpreter.invoke()
        let outputTensor = try interpreter.output(at: 0)
        print("Output Shape: \(outputTensor.shape)")

        let tokens = dataToInt32Array(outputTensor.data)
        var sentence = ""
        if let tokens = tokens {
            for token in tokens {
                if token == 50256 {
                    break
                }
                let word = whisper_token_to_str2(token)
                sentence += String(cString: word!)
            }
        }
        print(sentence)

    } catch {
        print("Transcription failed: \(error.localizedDescription)")
        return
    }
}
