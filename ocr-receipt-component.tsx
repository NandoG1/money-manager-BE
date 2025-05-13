import React, { useState, useRef } from 'react';

interface ReceiptAnalysis {
  judul: string;
  tanggal: string;
  subtotal: string;
}

const OCRReceiptComponent: React.FC = () => {
  const [image, setImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [ocrText, setOcrText] = useState<string>('');
  const [analysis, setAnalysis] = useState<ReceiptAnalysis | null>(null);
  const [error, setError] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Check if the file is an image
    if (!file.type.match('image.*')) {
      setError('Please select an image file');
      return;
    }

    // Read the file as a data URL
    const reader = new FileReader();
    reader.onload = (event) => {
      setImage(event.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    
    const file = e.dataTransfer.files?.[0];
    if (!file) return;

    // Check if the file is an image
    if (!file.type.match('image.*')) {
      setError('Please select an image file');
      return;
    }

    // Read the file as a data URL
    const reader = new FileReader();
    reader.onload = (event) => {
      setImage(event.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleSubmit = async () => {
    if (!image) {
      setError('Please select an image first');
      return;
    }

    setIsLoading(true);
    setError('');
    setOcrText('');
    setAnalysis(null);

    try {
      const response = await fetch('http://localhost:5000/api/ocr-receipt', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to process receipt');
      }

      const data = await response.json();
      setOcrText(data.ocr_text);
      
      try {
        // Parse the JSON string from the analysis field
        const analysisData = JSON.parse(data.analysis);
        setAnalysis(analysisData);
      } catch (jsonError) {
        console.error('Error parsing analysis JSON:', jsonError);
        setError('Error parsing receipt analysis');
      }
    } catch (error) {
      console.error('Error processing receipt:', error);
      setError(error instanceof Error ? error.message : 'An unexpected error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setImage(null);
    setOcrText('');
    setAnalysis(null);
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Receipt Scanner</h1>

      <div 
        className="border-2 border-dashed border-gray-300 rounded-lg p-6 mb-4 text-center cursor-pointer"
        onClick={() => fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <input 
          type="file" 
          accept="image/*" 
          onChange={handleFileChange} 
          className="hidden" 
          ref={fileInputRef}
        />
        
        {image ? (
          <div>
            <img 
              src={image} 
              alt="Receipt" 
              className="max-h-64 mx-auto mb-2" 
            />
            <p className="text-sm text-gray-500">Click to change image</p>
          </div>
        ) : (
          <div>
            <svg 
              className="mx-auto h-12 w-12 text-gray-400" 
              stroke="currentColor" 
              fill="none" 
              viewBox="0 0 48 48" 
              aria-hidden="true"
            >
              <path 
                d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" 
                strokeWidth={2} 
                strokeLinecap="round" 
                strokeLinejoin="round" 
              />
            </svg>
            <p className="mt-1 text-sm text-gray-600">
              Click to upload or drag and drop
            </p>
            <p className="text-xs text-gray-500">
              PNG, JPG, GIF up to 10MB
            </p>
          </div>
        )}
      </div>

      <div className="flex space-x-2 mb-4">
        <button
          onClick={handleSubmit}
          disabled={!image || isLoading}
          className="bg-blue-500 text-white px-4 py-2 rounded disabled:bg-blue-300 flex-1"
        >
          {isLoading ? 'Processing...' : 'Scan Receipt'}
        </button>
        
        <button
          onClick={handleReset}
          className="bg-gray-200 text-gray-800 px-4 py-2 rounded"
        >
          Reset
        </button>
      </div>

      {error && (
        <div className="bg-red-100 text-red-700 p-3 rounded mb-4">
          {error}
        </div>
      )}

      {analysis && (
        <div className="bg-gray-100 p-4 rounded-lg mb-4">
          <h2 className="text-xl font-semibold mb-2">Receipt Analysis</h2>
          <div className="grid grid-cols-2 gap-2">
            <div className="font-medium">Title:</div>
            <div>{analysis.judul}</div>
            <div className="font-medium">Date:</div>
            <div>{analysis.tanggal}</div>
            <div className="font-medium">Total Amount:</div>
            <div>{analysis.subtotal}</div>
          </div>
        </div>
      )}

      {ocrText && (
        <div className="mt-4">
          <h3 className="text-lg font-semibold mb-2">Raw OCR Text</h3>
          <div className="bg-gray-50 p-3 rounded border border-gray-200 text-sm font-mono whitespace-pre-line">
            {ocrText}
          </div>
        </div>
      )}
    </div>
  );
};

export default OCRReceiptComponent; 