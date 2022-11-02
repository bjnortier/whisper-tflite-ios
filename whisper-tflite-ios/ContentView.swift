//
//  ContentView.swift
//  whisper-tflite-ios
//
//  Created by Ben Nortier on 2022/11/02.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        VStack {
            Button(action: { transcribe() }) {
                Text("Transcribe")
            }
        }
        .padding()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
