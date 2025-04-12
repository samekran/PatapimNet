import UploadForm from "@/components/upload-form"

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8 bg-gradient-to-br from-green-50 to-blue-50">
      <div className="w-full max-w-4xl">
        <h1 className="text-5xl font-bold mb-8 text-center text-green-800 drop-shadow-md">Plant Health Analyzer</h1>
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
          <div className="p-8 space-y-8">
            <UploadForm />
          </div>
        </div>
      </div>
    </main>
  )
}
