
import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, FileText, Download, Printer, Loader2, CheckCircle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";

const ReportGeneration = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const printRef = useRef<HTMLDivElement>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [reportGenerated, setReportGenerated] = useState(false);
  const [formData, setFormData] = useState({
    vin: "",
    make: "",
    model: "",
    year: "",
    problemDescription: "",
    diagnosticFindings: "",
    partsNeeded: "",
    laborHours: "",
    estimatedCost: ""
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
    
    setReportGenerated(false);
  };

  const handleGenerate = async () => {
    
    if (!formData.vin || !formData.make || !formData.model || !formData.problemDescription) {
      toast({
        title: "Missing Information",
        description: "Please fill in VIN, Make, Model, and Problem Description.",
        variant: "destructive"
      });
      return;
    }

    setIsGenerating(true);
    
    
    setTimeout(() => {
      setIsGenerating(false);
      setReportGenerated(true);
      toast({
        title: "Report Generated",
        description: "Your service report has been created successfully.",
      });
    }, 1500);
  };

  const handleDownload = () => {
    
    const reportContent = `
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Service Report - ${formData.vin}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
        }
        .header {
            text-align: center;
            border-bottom: 3px solid #f97316;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #f97316;
            margin: 0;
            font-size: 28px;
        }
        .header p {
            color: #666;
            margin: 5px 0 0 0;
        }
        .section {
            margin-bottom: 25px;
        }
        .section-title {
            font-size: 18px;
            font-weight: bold;
            color: #333;
            border-bottom: 2px solid #ddd;
            padding-bottom: 8px;
            margin-bottom: 12px;
        }
        .info-row {
            display: flex;
            margin-bottom: 10px;
        }
        .info-label {
            font-weight: bold;
            min-width: 150px;
            color: #555;
        }
        .info-value {
            color: #333;
        }
        .description {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
        }
        .total-section {
            border-top: 3px solid #f97316;
            padding-top: 20px;
            margin-top: 30px;
            text-align: right;
        }
        .total-amount {
            font-size: 24px;
            font-weight: bold;
            color: #f97316;
        }
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #999;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Professional Service Report</h1>
        <p>Diagnostic & Repair Documentation</p>
        <p>Generated: ${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString()}</p>
    </div>

    <div class="section">
        <div class="section-title">Vehicle Information</div>
        <div class="info-row">
            <span class="info-label">VIN Number:</span>
            <span class="info-value">${formData.vin}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Vehicle:</span>
            <span class="info-value">${formData.year} ${formData.make} ${formData.model}</span>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Problem Description</div>
        <div class="description">${formData.problemDescription}</div>
    </div>

    ${formData.diagnosticFindings ? `
    <div class="section">
        <div class="section-title">Diagnostic Findings</div>
        <div class="description">${formData.diagnosticFindings}</div>
    </div>
    ` : ''}

    ${formData.partsNeeded ? `
    <div class="section">
        <div class="section-title">Parts Needed</div>
        <div class="description">${formData.partsNeeded}</div>
    </div>
    ` : ''}

    <div class="section">
        <div class="section-title">Service Details</div>
        ${formData.laborHours ? `
        <div class="info-row">
            <span class="info-label">Labor Hours:</span>
            <span class="info-value">${formData.laborHours} hours</span>
        </div>
        ` : ''}
        ${formData.estimatedCost ? `
        <div class="total-section">
            <span class="info-label">Estimated Total Cost:</span>
            <div class="total-amount">$${parseFloat(formData.estimatedCost).toFixed(2)}</div>
        </div>
        ` : ''}
    </div>

    <div class="footer">
        <p>This is an official service report generated by the Vehicle Intelligence System</p>
        <p>© ${new Date().getFullYear()} - All Rights Reserved</p>
    </div>
</body>
</html>
    `;

    
    const blob = new Blob([reportContent], { type: 'text/html' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `service-report-${formData.vin || 'draft'}-${Date.now()}.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);

    toast({
      title: "Report Downloaded",
      description: "Your report has been saved as an HTML file.",
    });
  };

  const handlePrint = () => {
    if (!reportGenerated) {
      toast({
        title: "Generate Report First",
        description: "Please generate the report before printing.",
        variant: "destructive"
      });
      return;
    }

   
    const printWindow = window.open('', '_blank');
    if (printWindow) {
      printWindow.document.write(`
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Service Report - ${formData.vin}</title>
    <style>
        @media print {
            body { margin: 0; padding: 20px; }
        }
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
        }
        .header {
            text-align: center;
            border-bottom: 3px solid #f97316;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #f97316;
            margin: 0;
            font-size: 28px;
        }
        .section {
            margin-bottom: 25px;
            page-break-inside: avoid;
        }
        .section-title {
            font-size: 18px;
            font-weight: bold;
            color: #333;
            border-bottom: 2px solid #ddd;
            padding-bottom: 8px;
            margin-bottom: 12px;
        }
        .info-row {
            margin-bottom: 10px;
        }
        .info-label {
            font-weight: bold;
            display: inline-block;
            min-width: 150px;
        }
        .description {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
        }
        .total-section {
            border-top: 3px solid #f97316;
            padding-top: 20px;
            margin-top: 30px;
            text-align: right;
        }
        .total-amount {
            font-size: 24px;
            font-weight: bold;
            color: #f97316;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Professional Service Report</h1>
        <p>Generated: ${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString()}</p>
    </div>

    <div class="section">
        <div class="section-title">Vehicle Information</div>
        <div class="info-row">
            <span class="info-label">VIN Number:</span>
            <span>${formData.vin}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Vehicle:</span>
            <span>${formData.year} ${formData.make} ${formData.model}</span>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Problem Description</div>
        <div class="description">${formData.problemDescription}</div>
    </div>

    ${formData.diagnosticFindings ? `
    <div class="section">
        <div class="section-title">Diagnostic Findings</div>
        <div class="description">${formData.diagnosticFindings}</div>
    </div>
    ` : ''}

    ${formData.partsNeeded ? `
    <div class="section">
        <div class="section-title">Parts Needed</div>
        <div class="description">${formData.partsNeeded}</div>
    </div>
    ` : ''}

    ${formData.laborHours || formData.estimatedCost ? `
    <div class="section">
        <div class="section-title">Service Details</div>
        ${formData.laborHours ? `<div class="info-row"><span class="info-label">Labor Hours:</span> ${formData.laborHours} hours</div>` : ''}
        ${formData.estimatedCost ? `
        <div class="total-section">
            <span class="info-label">Estimated Total Cost:</span>
            <div class="total-amount">$${parseFloat(formData.estimatedCost).toFixed(2)}</div>
        </div>
        ` : ''}
    </div>
    ` : ''}
</body>
</html>
      `);
      printWindow.document.close();
      
      
      setTimeout(() => {
        printWindow.print();
      }, 250);
    }

    toast({
      title: "Print Dialog Opened",
      description: "Your report is ready to print.",
    });
  };

  const isFormValid = formData.vin && formData.make && formData.model && formData.problemDescription;

  return (
    <div className="min-h-screen bg-gradient-hero">
      {/* Header */}
      <header className="border-b border-border glass-card">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <Button
              variant="ghost"
              onClick={() => navigate("/dashboard/mechanic")}
              className="gap-2"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Dashboard
            </Button>
            <h1 className="text-xl font-bold text-gradient-mechanic">Report Generation</h1>
            <div className="w-24" />
          </div>
        </div>
      </header>

      
      <div className="container mx-auto px-4 py-12">
        <div className="grid md:grid-cols-2 gap-8 max-w-7xl mx-auto">
         
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
          >
            <Card className="glass-card p-8">
              <div className="flex items-center gap-3 mb-8">
                <div className="w-12 h-12 rounded-xl bg-gradient-mechanic flex items-center justify-center">
                  <FileText className="w-6 h-6 text-background" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold">Service Report Form</h2>
                  <p className="text-sm text-muted-foreground">Fill in diagnostic details</p>
                </div>
              </div>

              <div className="space-y-6">
                
                <div className="space-y-4">
                  <h3 className="font-semibold text-lg">Vehicle Information</h3>
                  
                  <div>
                    <Label htmlFor="vin">VIN Number *</Label>
                    <Input
                      id="vin"
                      name="vin"
                      value={formData.vin}
                      onChange={handleChange}
                      placeholder="1HGBH41JXMN109186"
                      className="mt-1"
                    />
                  </div>

                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <Label htmlFor="make">Make *</Label>
                      <Input
                        id="make"
                        name="make"
                        value={formData.make}
                        onChange={handleChange}
                        placeholder="Toyota"
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label htmlFor="model">Model *</Label>
                      <Input
                        id="model"
                        name="model"
                        value={formData.model}
                        onChange={handleChange}
                        placeholder="Camry"
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label htmlFor="year">Year</Label>
                      <Input
                        id="year"
                        name="year"
                        value={formData.year}
                        onChange={handleChange}
                        placeholder="2023"
                        className="mt-1"
                      />
                    </div>
                  </div>
                </div>

                
                <div>
                  <Label htmlFor="problemDescription">Problem Description *</Label>
                  <Textarea
                    id="problemDescription"
                    name="problemDescription"
                    value={formData.problemDescription}
                    onChange={handleChange}
                    placeholder="Describe the issue the customer is experiencing..."
                    rows={4}
                    className="mt-1"
                  />
                </div>

                
                <div>
                  <Label htmlFor="diagnosticFindings">Diagnostic Findings</Label>
                  <Textarea
                    id="diagnosticFindings"
                    name="diagnosticFindings"
                    value={formData.diagnosticFindings}
                    onChange={handleChange}
                    placeholder="Enter your diagnostic results and findings..."
                    rows={4}
                    className="mt-1"
                  />
                </div>

                
                <div className="space-y-4">
                  <h3 className="font-semibold text-lg">Service Details</h3>
                  
                  <div>
                    <Label htmlFor="partsNeeded">Parts Needed</Label>
                    <Textarea
                      id="partsNeeded"
                      name="partsNeeded"
                      value={formData.partsNeeded}
                      onChange={handleChange}
                      placeholder="List required parts with part numbers..."
                      rows={3}
                      className="mt-1"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="laborHours">Labor Hours</Label>
                      <Input
                        id="laborHours"
                        name="laborHours"
                        type="number"
                        step="0.5"
                        value={formData.laborHours}
                        onChange={handleChange}
                        placeholder="3.5"
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label htmlFor="estimatedCost">Estimated Cost ($)</Label>
                      <Input
                        id="estimatedCost"
                        name="estimatedCost"
                        type="number"
                        step="0.01"
                        value={formData.estimatedCost}
                        onChange={handleChange}
                        placeholder="450.00"
                        className="mt-1"
                      />
                    </div>
                  </div>
                </div>

               
                <Button
                  onClick={handleGenerate}
                  disabled={isGenerating || !isFormValid}
                  className="w-full bg-gradient-mechanic hover:opacity-90 text-white font-semibold py-6 text-lg"
                >
                  {isGenerating ? (
                    <>
                      <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                      Generating Report...
                    </>
                  ) : reportGenerated ? (
                    <>
                      <CheckCircle className="w-5 h-5 mr-2" />
                      Report Generated
                    </>
                  ) : (
                    <>
                      <FileText className="w-5 h-5 mr-2" />
                      Generate Report
                    </>
                  )}
                </Button>
                {!isFormValid && (
                  <p className="text-xs text-muted-foreground text-center">
                    * Required fields must be filled
                  </p>
                )}
              </div>
            </Card>
          </motion.div>

          
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
          >
            <Card className="glass-card p-8 sticky top-8">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold">Report Preview</h2>
                {reportGenerated && (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="flex items-center gap-2 text-green-500"
                  >
                    <CheckCircle className="w-5 h-5" />
                    <span className="text-sm font-semibold">Ready</span>
                  </motion.div>
                )}
              </div>
              
              <div ref={printRef} className="bg-muted/20 rounded-lg p-6 min-h-[500px] mb-6">
                <AnimatePresence mode="wait">
                  {isFormValid ? (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                    >
                      <div className="text-center mb-8">
                        <h3 className="text-2xl font-bold mb-2">Service Report</h3>
                        <p className="text-sm text-muted-foreground">Professional Diagnostic Report</p>
                        <p className="text-xs text-muted-foreground mt-1">
                          Generated: {new Date().toLocaleDateString()}
                        </p>
                      </div>

                      <div className="space-y-4 text-sm">
                        <div className="pb-3 border-b border-border">
                          <span className="font-semibold text-xs text-muted-foreground uppercase tracking-wide">VIN Number</span>
                          <p className="font-mono text-base mt-1">{formData.vin}</p>
                        </div>
                        
                        <div className="pb-3 border-b border-border">
                          <span className="font-semibold text-xs text-muted-foreground uppercase tracking-wide">Vehicle</span>
                          <p className="text-base mt-1">{formData.year} {formData.make} {formData.model}</p>
                        </div>
                        
                        <div className="pb-3 border-b border-border">
                          <span className="font-semibold text-xs text-muted-foreground uppercase tracking-wide">Problem Description</span>
                          <p className="text-muted-foreground mt-2 leading-relaxed">{formData.problemDescription}</p>
                        </div>
                        
                        {formData.diagnosticFindings && (
                          <div className="pb-3 border-b border-border">
                            <span className="font-semibold text-xs text-muted-foreground uppercase tracking-wide">Diagnostic Findings</span>
                            <p className="text-muted-foreground mt-2 leading-relaxed">{formData.diagnosticFindings}</p>
                          </div>
                        )}
                        
                        {formData.partsNeeded && (
                          <div className="pb-3 border-b border-border">
                            <span className="font-semibold text-xs text-muted-foreground uppercase tracking-wide">Parts Needed</span>
                            <p className="text-muted-foreground mt-2 leading-relaxed">{formData.partsNeeded}</p>
                          </div>
                        )}
                        
                        {(formData.laborHours || formData.estimatedCost) && (
                          <div className="pt-4">
                            {formData.laborHours && (
                              <div className="flex justify-between mb-2">
                                <span className="text-muted-foreground">Labor Hours:</span>
                                <span className="font-semibold">{formData.laborHours} hrs</span>
                              </div>
                            )}
                            {formData.estimatedCost && (
                              <div className="flex justify-between items-center pt-3 border-t-2 border-accent-mechanic">
                                <span className="font-semibold text-base">Total Estimated Cost:</span>
                                <span className="font-bold text-2xl text-accent-mechanic">
                                  ${parseFloat(formData.estimatedCost).toFixed(2)}
                                </span>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </motion.div>
                  ) : (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="flex flex-col items-center justify-center h-[500px] text-muted-foreground"
                    >
                      <FileText className="w-16 h-16 mb-4 opacity-30" />
                      <p className="text-center">Fill out the form to preview report</p>
                      <p className="text-xs text-center mt-2">Required: VIN, Make, Model, Problem Description</p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              
              <div className="space-y-3">
                <Button 
                  onClick={handleDownload}
                  disabled={!reportGenerated}
                  className="w-full gap-2 bg-secondary hover:bg-secondary/90"
                >
                  <Download className="w-4 h-4 text-black" />
                  Download as HTML
                </Button>
                <Button 
                  onClick={handlePrint}
                  disabled={!reportGenerated}
                  variant="outline" 
                  className="w-full gap-2"
                >
                  <Printer className="w-4 h-4" />
                  Print Report
                </Button>
              </div>
              
              {!reportGenerated && isFormValid && (
                <p className="text-xs text-center text-muted-foreground mt-4">
                  Click "Generate Report" to enable download and print
                </p>
              )}
            </Card>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default ReportGeneration;