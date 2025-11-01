import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';

/**
 * Generate a PDF from slide elements
 * @param slideElements - Array of HTML elements representing slides
 * @param filename - Name of the PDF file to download
 * @param onProgress - Optional callback for progress updates
 */
export const generateSlidePDF = async (
  slideElements: HTMLElement[],
  filename: string,
  onProgress?: (current: number, total: number) => void
): Promise<void> => {
  if (slideElements.length === 0) {
    throw new Error('No slides to export');
  }

  // PDF dimensions for 16:9 aspect ratio (standard presentation)
  const pdfWidth = 297; // A4 landscape width in mm
  const pdfHeight = 167.06; // 16:9 ratio height in mm
  
  const pdf = new jsPDF({
    orientation: 'landscape',
    unit: 'mm',
    format: [pdfWidth, pdfHeight]
  });

  for (let i = 0; i < slideElements.length; i++) {
    const element = slideElements[i];
    
    // Report progress
    if (onProgress) {
      onProgress(i + 1, slideElements.length);
    }

    try {
      // Capture the slide as canvas
      const canvas = await html2canvas(element, {
        scale: 2, // Higher quality
        useCORS: true,
        logging: false,
        backgroundColor: null,
        windowWidth: 1600,
        windowHeight: 900
      });

      const imgData = canvas.toDataURL('image/png');
      
      // Add new page for slides after the first
      if (i > 0) {
        pdf.addPage([pdfWidth, pdfHeight], 'landscape');
      }

      // Add image to PDF
      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
    } catch (error) {
      console.error(`Error capturing slide ${i + 1}:`, error);
      throw new Error(`Failed to capture slide ${i + 1}`);
    }
  }

  // Save the PDF
  pdf.save(filename);
};

/**
 * Generate PDF from a single slide that needs to be rendered multiple times
 * @param renderSlide - Function that renders a slide at given index
 * @param totalSlides - Total number of slides
 * @param containerId - ID of container element
 * @param filename - Name of PDF file
 * @param onProgress - Progress callback
 */
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
    // Render the slide
    renderSlide(i);
    
    // Wait for render to complete
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
