"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { Upload, Leaf } from "lucide-react"
import { Button } from "@/components/ui/button"
import ResultsDisplay from "./results-display"

type AnalysisResult = {
  condition: string
  confidence: number
  heatmapImage: string
  recommendation: string
  infoReliability: number
}

export default function UploadForm() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]
      setFile(selectedFile)
      setAnalysisResults(null)
      // Create preview URL for the image
      const url = URL.createObjectURL(selectedFile)
      setPreviewUrl(url)
    }
  }

  // Add cleanup function for the preview URL
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [previewUrl])

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!file) return

    setIsAnalyzing(true)
    try {
      const formData = new FormData()
      formData.append('image', file)

      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      })

      // First try to get the response as text
      const responseText = await response.text()
      let results
      
      try {
        // Then try to parse it as JSON
        results = JSON.parse(responseText)
      } catch (e) {
        console.error('Server response:', responseText)
        throw new Error('Server returned invalid JSON. Check the console for details.')
      }

      if (!response.ok) {
        throw new Error(results.error || 'Analysis failed')
      }

      setAnalysisResults(results)
    } catch (error) {
      console.error('Error analyzing image:', error)
      alert(error instanceof Error ? error.message : 'An error occurred during analysis')
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="flex items-center justify-center w-full">
          {previewUrl ? (
            <div className="relative w-full h-64">
              <img
                src={previewUrl}
                alt="Preview"
                className="w-full h-full object-contain rounded-2xl"
              />
              <label
                htmlFor="dropzone-file"
                className="absolute inset-0 flex flex-col items-center justify-center opacity-0 hover:opacity-100 bg-green-50/90 transition-opacity duration-300 cursor-pointer rounded-2xl"
              >
                <Upload className="w-12 h-12 mb-3 text-green-500" />
                <p className="text-sm text-green-700">
                  <span className="font-semibold">Click to replace</span>
                </p>
                <input
                  id="dropzone-file"
                  type="file"
                  className="hidden"
                  onChange={handleFileChange}
                  accept="image/png, image/jpeg, image/jpg"
                />
              </label>
            </div>
          ) : (
            <label
              htmlFor="dropzone-file"
              className="flex flex-col items-center justify-center w-full h-64 border-2 border-green-300 border-dashed rounded-2xl cursor-pointer bg-green-50 hover:bg-green-100 transition-colors duration-300"
            >
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <Upload className="w-12 h-12 mb-3 text-green-500" />
                <p className="mb-2 text-sm text-green-700">
                  <span className="font-semibold">Click to upload</span> or drag and drop
                </p>
                <p className="text-xs text-green-600">PNG, JPG or JPEG (MAX. 800x400px)</p>
              </div>
              <input
                id="dropzone-file"
                type="file"
                className="hidden"
                onChange={handleFileChange}
                accept="image/png, image/jpeg, image/jpg"
              />
            </label>
          )}
        </div>
        {file && (
          <div className="text-center">
            <p className="text-sm text-green-600 mb-2">Selected file: {file.name}</p>
            <Button
              type="submit"
              className="bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-6 rounded-full"
              disabled={isAnalyzing}
            >
              <Leaf className="w-5 h-5 mr-2" />
              {isAnalyzing ? 'Analyzing...' : 'Analyze Plant'}
            </Button>
          </div>
        )}
      </form>
      <ResultsDisplay show={!!analysisResults} results={analysisResults} />
    </div>
  )
}
