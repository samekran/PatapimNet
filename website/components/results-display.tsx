"use client"

import Image from "next/image"
import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { AlertTriangle, ThumbsUp, Droplet, Sun, Wind } from "lucide-react"

type AnalysisResult = {
  species: string
  speciesConfidence: number
  speciesHeatmap: string
  condition: string
  confidence: number
  heatmapImage: string
  recommendation: string
  infoReliability: number
}

export default function ResultsDisplay({ 
  show = false,
  results
}: { 
  show?: boolean
  results: AnalysisResult | null 
}) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted || !show || !results) return null

  const getStatusIcon = (condition: string) => {
    return condition === "Healthy" ? (
      <ThumbsUp className="w-8 h-8 text-green-500" />
    ) : (
      <AlertTriangle className="w-8 h-8 text-yellow-500" />
    )
  }

  return (
    <div className="space-y-8">
      <h2 className="text-3xl font-semibold text-green-800 text-center">Analysis Results</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <Card className="bg-gradient-to-br from-blue-100 to-green-100 shadow-md">
          <CardHeader>
            <CardTitle>Plant Species</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-green-800 mb-4">{results.species}</p>
            <div>
              <p className="text-sm text-green-700 mb-1">Confidence Level</p>
              <Progress value={results.speciesConfidence} className="w-full h-3 bg-green-200" />
              <p className="text-sm text-green-700 mt-1 text-right">{results.speciesConfidence}%</p>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-gradient-to-br from-purple-100 to-pink-100 shadow-md">
          <CardHeader>
            <CardTitle>Species Analysis Regions</CardTitle>
          </CardHeader>
          <CardContent>
            <Image
              src={results.speciesHeatmap || "/placeholder.svg"}
              alt="Species analysis heatmap"
              width={400}
              height={300}
              className="w-full h-auto rounded-lg shadow-sm"
            />
          </CardContent>
        </Card>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="bg-gradient-to-br from-green-100 to-blue-100 shadow-md">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Plant Condition</span>
              {getStatusIcon(results.condition)}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-green-800 mb-4">{results.condition}</p>
            <div>
              <p className="text-sm text-green-700 mb-1">Confidence Level</p>
              <Progress value={results.confidence} className="w-full h-3 bg-green-200" />
              <p className="text-sm text-green-700 mt-1 text-right">{results.confidence}%</p>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-gradient-to-br from-purple-100 to-pink-100 shadow-md">
          <CardHeader>
            <CardTitle>Regions of Interest</CardTitle>
          </CardHeader>
          <CardContent>
            <Image
              src={results.heatmapImage || "/placeholder.svg"}
              alt="Analysis heatmap"
              width={400}
              height={300}
              className="w-full h-auto rounded-lg shadow-sm"
            />
          </CardContent>
        </Card>
        <Card className="md:col-span-2 bg-gradient-to-br from-yellow-100 to-orange-100 shadow-md">
          <CardHeader>
            <CardTitle className="flex items-center">
              <span>Care Recommendations</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-lg text-green-800">{results.recommendation}</p>
            <div className="flex justify-around mt-4">
              <div className="flex flex-col items-center">
                <Droplet className="w-8 h-8 text-blue-500" />
                <span className="text-sm mt-1">Water</span>
              </div>
              <div className="flex flex-col items-center">
                <Sun className="w-8 h-8 text-yellow-500" />
                <span className="text-sm mt-1">Sunlight</span>
              </div>
              <div className="flex flex-col items-center">
                <Wind className="w-8 h-8 text-green-500" />
                <span className="text-sm mt-1">Air</span>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="md:col-span-2 bg-gradient-to-br from-blue-100 to-green-100 shadow-md">
          <CardHeader>
            <CardTitle>Retrieved Information Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-4">
              <div className="w-0 h-0 border-l-[40px] border-l-transparent border-b-[80px] border-b-blue-500 border-r-[40px] border-r-transparent relative">
                <span className="absolute top-14 left-1/2 transform -translate-x-1/2 text-white font-bold">
                  {results.infoReliability}%
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-lg font-semibold text-blue-800">Information Reliability Score</span>
                <span className="text-sm text-blue-600">Based on data quality and relevance</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
