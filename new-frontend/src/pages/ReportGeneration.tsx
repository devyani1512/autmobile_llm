import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, FileText, Download, Mail, Printer, Loader2 } from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";

const ReportGeneration = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [isGenerating, setIsGenerating] = useState(false);
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
  };

  const handleGenerate = async () => {
    setIsGenerating(true);
    
    // Simulate report generation
    setTimeout(() => {
      setIsGenerating(false);
      toast({
        title: "Report Generated",
        description: "Your service report has been created successfully.",
      });
    }, 2000);
  };

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

      {/* Main Content */}
      <div className="container mx-auto px-4 py-12">
        <div className="grid md:grid-cols-2 gap-8 max-w-7xl mx-auto">
          {/* Form */}
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
                <h2 className="text-2xl font-bold">Service Report Form</h2>
              </div>

              <div className="space-y-6">
                {/* Vehicle Information */}
                <div className="space-y-4">
                  <h3 className="font-semibold text-lg">Vehicle Information</h3>
                  
                  <div>
                    <Label htmlFor="vin">VIN Number</Label>
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
                      <Label htmlFor="make">Make</Label>
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
                      <Label htmlFor="model">Model</Label>
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

                {/* Problem Description */}
                <div>
                  <Label htmlFor="problemDescription">Problem Description</Label>
                  <Textarea
                    id="problemDescription"
                    name="problemDescription"
                    value={formData.problemDescription}
                    onChange={handleChange}
                    placeholder="Describe the issue..."
                    rows={4}
                    className="mt-1"
                  />
                </div>

                {/* Diagnostic Findings */}
                <div>
                  <Label htmlFor="diagnosticFindings">Diagnostic Findings</Label>
                  <Textarea
                    id="diagnosticFindings"
                    name="diagnosticFindings"
                    value={formData.diagnosticFindings}
                    onChange={handleChange}
                    placeholder="Enter diagnostic results..."
                    rows={4}
                    className="mt-1"
                  />
                </div>

                {/* Service Details */}
                <div className="space-y-4">
                  <h3 className="font-semibold text-lg">Service Details</h3>
                  
                  <div>
                    <Label htmlFor="partsNeeded">Parts Needed</Label>
                    <Textarea
                      id="partsNeeded"
                      name="partsNeeded"
                      value={formData.partsNeeded}
                      onChange={handleChange}
                      placeholder="List parts..."
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
                        value={formData.estimatedCost}
                        onChange={handleChange}
                        placeholder="450.00"
                        className="mt-1"
                      />
                    </div>
                  </div>
                </div>

                {/* Generate Button */}
                <Button
                  onClick={handleGenerate}
                  disabled={isGenerating}
                  className="w-full bg-gradient-mechanic hover:opacity-90 text-background font-semibold py-6 text-lg"
                >
                  {isGenerating ? (
                    <>
                      <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                      Generating Report...
                    </>
                  ) : (
                    <>
                      <FileText className="w-5 h-5 mr-2" />
                      Generate Report
                    </>
                  )}
                </Button>
              </div>
            </Card>
          </motion.div>

          {/* Preview */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
          >
            <Card className="glass-card p-8 sticky top-8">
              <h2 className="text-2xl font-bold mb-6">Report Preview</h2>
              
              <div className="bg-muted/20 rounded-lg p-6 min-h-[500px] mb-6">
                <div className="text-center mb-8">
                  <h3 className="text-2xl font-bold mb-2">Service Report</h3>
                  <p className="text-sm text-muted-foreground">Professional Diagnostic Report</p>
                </div>

                {formData.vin && (
                  <div className="space-y-4 text-sm">
                    <div>
                      <span className="font-semibold">VIN:</span> {formData.vin}
                    </div>
                    {(formData.make || formData.model || formData.year) && (
                      <div>
                        <span className="font-semibold">Vehicle:</span> {formData.year} {formData.make} {formData.model}
                      </div>
                    )}
                    {formData.problemDescription && (
                      <div>
                        <span className="font-semibold">Problem:</span>
                        <p className="text-muted-foreground mt-1">{formData.problemDescription}</p>
                      </div>
                    )}
                    {formData.diagnosticFindings && (
                      <div>
                        <span className="font-semibold">Findings:</span>
                        <p className="text-muted-foreground mt-1">{formData.diagnosticFindings}</p>
                      </div>
                    )}
                    {formData.estimatedCost && (
                      <div className="pt-4 border-t border-border">
                        <span className="font-semibold text-lg">Total: ${formData.estimatedCost}</span>
                      </div>
                    )}
                  </div>
                )}

                {!formData.vin && (
                  <div className="flex items-center justify-center h-64 text-muted-foreground">
                    Fill out the form to preview report
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="space-y-3">
                <Button variant="outline" className="w-full gap-2">
                  <Download className="w-4 h-4" />
                  Download PDF
                </Button>
                <Button variant="outline" className="w-full gap-2">
                  <Printer className="w-4 h-4" />
                  Print Report
                </Button>
                <Button variant="outline" className="w-full gap-2">
                  <Mail className="w-4 h-4" />
                  Email Report
                </Button>
              </div>
            </Card>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default ReportGeneration;
