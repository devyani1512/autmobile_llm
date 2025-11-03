import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface QueryRequest {
  query: string;
  mode: 'owner' | 'mechanic';
}

interface QueryResponse {
  answer: string;
}

export const askQuestion = async (
  query: string, 
  mode: 'owner' | 'mechanic'
): Promise<QueryResponse> => {
  try {
    const response = await axios.post<QueryResponse>(`${API_URL}/api/ask`, {
      query,
      mode
    });
    return response.data;
  } catch (error) {
    console.error('Error asking question:', error);
    throw error;
  }
};

export const uploadPDF = async (file: File) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await axios.post(`${API_URL}/api/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error uploading PDF:', error);
    throw error;
  }
};

export const getAssetURL = (filename: string): string => {
  return `${API_URL}/assets/${filename}`;
};