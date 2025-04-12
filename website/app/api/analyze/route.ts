import { NextResponse } from 'next/server'
import { writeFile } from 'fs/promises'
import { exec } from 'child_process'
import { promisify } from 'util'
import path from 'path'
import os from 'os'
import { unlink } from 'fs/promises'

console.log('>>> analyze route loaded');
export const runtime = 'nodejs';

const execAsync = promisify(exec)

export async function POST(request: Request) {
  try {
    const formData = await request.formData()
    const image = formData.get('image') as File

    if (!image) {
      return NextResponse.json(
        { error: 'No image provided' },
        { status: 400 }
      )
    }

    // Validate image type
    if (!image.type.startsWith('image/')) {
      return NextResponse.json(
        { error: 'Invalid file type. Please upload an image.' },
        { status: 400 }
      )
    }

    // Add size validation if needed
    const MAX_SIZE = 5 * 1024 * 1024 // 5MB
    if (image.size > MAX_SIZE) {
      return NextResponse.json(
        { error: 'Image size too large. Please upload an image smaller than 5MB.' },
        { status: 400 }
      )
    }

    // Create a temporary file for the uploaded image
    const bytes = await image.arrayBuffer()
    const buffer = Buffer.from(bytes)
    
    // Create a temporary file with a unique name
    const tempDir = os.tmpdir()
    const tempImagePath = path.join(tempDir, `upload-${Date.now()}.jpg`)
    
    try {
      await writeFile(tempImagePath, buffer)
      
      // Get the absolute path to predict.py
      const scriptPath = path.join(process.cwd(), 'app', 'api', 'analyze', 'predict.py')
      
      // Run the Python script with the image path
      const pythonPath = '/usr/local/bin/python3'  // Update this with your actual Python path
      const { stdout, stderr } = await execAsync(`"${pythonPath}" "${scriptPath}" "${tempImagePath}"`)
      
      if (stderr) {
        console.error('Python script error:', stderr)
        return NextResponse.json(
          { error: 'Model inference failed: ' + stderr },
          { status: 500 }
        )
      }

      try {
        const results = JSON.parse(stdout)
        return NextResponse.json(results)
      } catch (parseError) {
        console.error('Error parsing Python output:', stdout)
        return NextResponse.json(
          { error: 'Invalid model output' },
          { status: 500 }
        )
      }
      
    } catch (error) {
      console.error('Error running model:', error)
      return NextResponse.json(
        { error: 'Model inference failed: ' + (error as Error).message },
        { status: 500 }
      )
    } finally {
      try {
        await unlink(tempImagePath)
      } catch (error) {
        console.error('Error cleaning up temp file:', error)
      }
    }    
  } catch (error) {
    console.error('Error processing image:', error)
    return NextResponse.json(
      { error: 'Error processing image' },
      { status: 500 }
    )
  }
} 

// Add a GET handler to test if the route is accessible
export async function GET() {
  return NextResponse.json({ message: 'API is working' })
}