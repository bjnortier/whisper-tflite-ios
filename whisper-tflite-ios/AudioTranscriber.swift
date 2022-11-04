//
//  AudioTranscriber.swift
//  whisper-tflite-ios
//
//  Created by Ben Nortier on 2022/11/02.
//

import Foundation
import TensorFlowLite

extension String: LocalizedError {
    public var errorDescription: String? { return self }
}

private func dataToInt32Array(_ data: Data) -> [Int32]? {
    guard data.count % MemoryLayout<Int32>.stride == 0 else { return nil }

    return data.withUnsafeBytes { .init($0.bindMemory(to: Int32.self)) }
}

@_transparent @discardableResult public func measure(label: String? = nil, tests: Int = 1, printResults output: Bool = true, setup: @escaping () -> Void = {}, _ block: @escaping () -> Void) -> Double {
    guard tests > 0 else { fatalError("Number of tests must be greater than 0") }

    var avgExecutionTime: CFAbsoluteTime = 0
    for _ in 1 ... tests {
        setup()
        let start = CFAbsoluteTimeGetCurrent()
        block()
        let end = CFAbsoluteTimeGetCurrent()
        avgExecutionTime += end - start
    }

    avgExecutionTime /= CFAbsoluteTime(tests)

    if output {
        let avgTimeStr = "\(avgExecutionTime)".replacingOccurrences(of: "e|E", with: " × 10^", options: .regularExpression, range: nil)

        if let label = label {
            print(label, "▿")
            print("\tExecution time: \(avgTimeStr)s")
            print("\tNumber of tests: \(tests)\n")
        } else {
            print("Execution time: \(avgTimeStr)s")
            print("Number of tests: \(tests)\n")
        }
    }

    return avgExecutionTime
}

func transcribe() {
    measure(label: "transcribe()") {
        do {
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
            ) else {
                throw "whisper.tflite not found"
            }

            let interpreter = try Interpreter(modelPath: modelPath)
            try interpreter.allocateTensors()
            let inputShape = try interpreter.input(at: 0).shape
            print("Input Shape: \(inputShape)")

            let floatData = Array(UnsafeBufferPointer(start: result, count: 240000))

            let audioBufferData = floatData.withUnsafeBufferPointer(Data.init)
            print("Buffer size: \(audioBufferData.count)")

            try interpreter.copy(audioBufferData, toInputAt: 0)
            try interpreter.invoke()

            measure(label: "invoke()") {
                do {
                    try interpreter.invoke()
                } catch {}
            }
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
            print("Transcription failed: \(error)")
            return
        }
    }
}
