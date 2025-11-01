import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';

export const generateSlidesFromRenderer = async (
  renderSlide: (index: number) => void,
  totalSlides: number,
  containerId: string,
  filename: string,
  onProgress?: (current: number, total: number) => void
): Promise<void> => {
  const container = document.getElementById(containerId);
  if (!container) {
    throw new Error(`Container #${containerId} not found`);
  }

  const pdfWidth = 297;
  const pdfHeight = 167.06;
  
  const pdf = new jsPDF({
    orientation: 'landscape',
    unit: 'mm',
    format: [pdfWidth, pdfHeight]
  });

  for (let i = 0; i < totalSlides; i++) {
    renderSlide(i);
    
    await new Promise(resolve => setTimeout(resolve, 100));

    if (onProgress) {
      onProgress(i + 1, totalSlides);
    }

    try {
      const canvas = await html2canvas(container, {
        scale: 2,
        useCORS: true,
        logging: false,
        backgroundColor: null,
        windowWidth: 1600,
        windowHeight: 900
      });

      const imgData = canvas.toDataURL('image/png');
      
      if (i > 0) {
        pdf.addPage([pdfWidth, pdfHeight], 'landscape');
      }

      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
    } catch (error) {
      console.error(`Error capturing slide ${i + 1}:`, error);
      throw new Error(`Failed to capture slide ${i + 1}`);
    }
  }

  pdf.save(filename);
};
