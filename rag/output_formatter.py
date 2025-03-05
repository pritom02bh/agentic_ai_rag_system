import re
import logging
logger = logging.getLogger(__name__)
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
import numpy as np
from datetime import datetime

@dataclass
class InventoryItem:
    product_id: str
    name: str
    current_stock: int
    reorder_point: int
    lead_time: str
    supplier: str
    notes: str

@dataclass
class AnalyticsInsight:
    title: str
    description: str
    recommendation: str
    priority: str  # 'high', 'medium', 'low'
    metrics: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "description": self.description,
            "recommendation": self.recommendation,
            "priority": self.priority,
            "metrics": self.metrics
        }

@dataclass
class FormattedResponse:
    message_type: str  # 'inventory', 'transport', or 'general'
    summary: str
    structured_data: Optional[Dict] = None
    tables: Optional[List[Dict]] = None
    charts: Optional[List[Dict]] = None
    insights: Optional[List[AnalyticsInsight]] = None
    trends: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "message_type": self.message_type,
            "summary": self.summary,
            "structured_data": self.structured_data,
            "tables": self.tables,
            "charts": self.charts,
            "insights": [insight.to_dict() for insight in self.insights] if self.insights else None,
            "trends": self.trends
        }

class OutputFormatter:
    def __init__(self):
        self.client = OpenAI()
        self.formatting_prompt = """
        Format the following medical supply chain data into a clear, structured response.
        IMPORTANT: Only format the response if it contains actual medical supply chain data.
        If the input is a general message or greeting, return it as is without any formatting.

        For medical supply chain data, follow this structure:
        1. Start with "# Medical Supplies Inventory Summary"
        2. Include a "## Current Inventory Overview" section with a table
        3. Add a "## Key Observations" section
        4. Include a "## Actionable Insights" section
        5. End with a brief "## Conclusion"

        Raw data to format:
        {data}
        """

    def format_response(self, raw_response: str) -> FormattedResponse:
        """Format the raw RAG response into a structured, UI-friendly format"""
        try:
            # More comprehensive check for general messages
            text = raw_response.lower()
            is_general = (
                len(text.split()) < 10 or  # Short messages
                any(greeting in text for greeting in ['hi', 'hello', 'hey', 'greetings']) or  # Greetings
                not any(keyword in text for keyword in ['inventory', 'transport', 'stock', 'shipment', 'supply', 'medical'])  # No supply chain terms
            )
            
            if is_general:
                return self._format_general_response(raw_response)
            
            # Get AI to format the response
            formatted = self._get_formatted_response(raw_response)
            
            # Parse the formatted response
            message_type = "inventory" if "Inventory" in raw_response else "transport"
            
            # Create structured data from the formatted response
            structured_data = self._extract_structured_data(formatted)
            
            # Extract tables from the formatted response
            tables = self._extract_tables(formatted)
            
            # Get insights
            insights = self._extract_insights(formatted)
            
            # Return formatted response
            return FormattedResponse(
                message_type=message_type,
                summary=formatted,
                structured_data=structured_data,
                tables=tables,
                insights=insights
            )
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return FormattedResponse(
                message_type="error",
                summary=raw_response
            )

    def _get_formatted_response(self, raw_data: str) -> str:
        """Use OpenAI to format the response according to the template"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": """You are a medical supply chain data formatter. Your task is to format data into clear, structured reports.
IMPORTANT: If the input is a general message or greeting (like 'hi', 'hello', etc.), return it exactly as is without any formatting or additional content.
Only apply formatting when the input contains actual medical supply chain data."""},
                    {"role": "user", "content": self.formatting_prompt.format(data=raw_data)}
                ],
                temperature=0.3  # Low temperature for consistent formatting
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting formatted response: {e}")
            return raw_data

    def _extract_structured_data(self, formatted_text: str) -> Dict:
        """Extract structured data from the formatted response"""
        try:
            # Extract table data
            table_data = []
            in_table = False
            headers = []
            
            for line in formatted_text.split('\n'):
                if '|' in line:
                    parts = [part.strip() for part in line.split('|') if part.strip()]
                    if not in_table:
                        headers = parts
                        in_table = True
                    elif not all(c == '-' for c in line if c not in '|-'):  # Skip separator line
                        row = dict(zip(headers, parts))
                        table_data.append(row)
            
            return {"inventory_items": table_data}
        except Exception as e:
            logger.error(f"Error extracting structured data: {e}")
            return {}

    def _extract_tables(self, formatted_text: str) -> List[Dict]:
        """Extract tables from the formatted response"""
        try:
            tables = []
            current_table = None
            
            for line in formatted_text.split('\n'):
                if '|' in line:
                    parts = [part.strip() for part in line.split('|') if part.strip()]
                    if current_table is None:
                        current_table = {
                            "headers": parts,
                            "rows": []
                        }
                    elif not all(c == '-' for c in line if c not in '|-'):  # Skip separator line
                        row = dict(zip(current_table["headers"], parts))
                        current_table["rows"].append(row)
                elif current_table is not None and current_table["rows"]:
                    tables.append(current_table)
                    current_table = None
            
            if current_table is not None and current_table["rows"]:
                tables.append(current_table)
            
            return tables
        except Exception as e:
            logger.error(f"Error extracting tables: {e}")
            return []

    def _extract_insights(self, formatted_text: str) -> List[Dict]:
        """Extract insights from the formatted response"""
        try:
            insights = []
            current_section = None
            current_insight = None
            
            for line in formatted_text.split('\n'):
                line = line.strip()
                
                if line.startswith('## Key Observations'):
                    current_section = 'observations'
                elif line.startswith('## Actionable Insights'):
                    current_section = 'insights'
                elif line.startswith('## '):
                    current_section = None
                elif current_section == 'observations' and line.startswith('- '):
                    insights.append({
                        "title": "Observation",
                        "description": line[2:],
                        "type": "observation"
                    })
                elif current_section == 'insights' and line[0].isdigit():
                    insights.append({
                        "title": "Action Item",
                        "description": line[3:],
                        "type": "action",
                        "priority": "high" if "immediate" in line.lower() or "critical" in line.lower() else "medium"
                    })
            
            return insights
        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
            return []

    def _get_ai_analysis(self, data: str) -> Dict:
        """
        Get AI-powered analysis of the data
        """
        try:
            prompt = self.analysis_prompt.format(data=data)
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a supply chain analytics expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error getting AI analysis: {e}")
            return None

    def _format_inventory_response(self, raw_response: str) -> FormattedResponse:
        """
        Format inventory-specific responses with enhanced analytics
        """
        # Extract inventory items
        items = []
        lines = raw_response.split('\n')
        
        # Extract summary
        summary = ""
        for line in lines:
            if "Inventory Update Summary" in line:
                summary = "Current Inventory Status Overview"
                break

        # Parse inventory items from the markdown table
        in_table = False
        headers = []
        for line in lines:
            line = line.strip()
            if "|" in line:
                parts = [part.strip() for part in line.split("|") if part.strip()]
                if not in_table:
                    headers = parts
                    in_table = True
                elif not all(c == '-' for c in line if c not in '|-'):  # Skip separator line
                    try:
                        item_dict = dict(zip(headers, parts))
                        current_stock = int(item_dict.get('Current Stock', '0').split()[0].replace(',', ''))
                        reorder_point = int(item_dict.get('Reorder Point', '0').split()[0].replace(',', ''))
                        
                        item = {
                            "product_id": item_dict.get('Product ID', '').strip(),
                            "name": item_dict.get('Item Description', '').strip(),
                            "current_stock": current_stock,
                            "reorder_point": reorder_point,
                            "lead_time": item_dict.get('Lead Time', '').strip(),
                            "supplier": item_dict.get('Supplier', '').strip(),
                            "location": item_dict.get('Storage Location', '').strip(),
                            "status": self._determine_status(current_stock, reorder_point)
                        }
                        items.append(item)
                    except (IndexError, ValueError, KeyError) as e:
                        print(f"Error parsing line: {e}")
                        continue

        if not items:
            return FormattedResponse(
                message_type="inventory",
                summary="No inventory data found in the response."
            )

        # Get AI analysis
        analysis = self._get_ai_analysis(json.dumps({
            "items": items,
            "raw_insights": raw_response
        }))
        
        if analysis:
            insights = [
                AnalyticsInsight(
                    title=insight['title'],
                    description=insight['description'],
                    recommendation=insight['recommendation'],
                    priority=insight['priority'],
                    metrics=insight.get('metrics', {})
                )
                for insight in analysis.get('insights', [])
            ]
        else:
            insights = []

        # Create enhanced visualizations
        charts = self._create_enhanced_charts(items, analysis)

        # Calculate trends and patterns
        trends = self._calculate_trends(items)

        # Create table structure with enhanced metrics
        table = self._create_enhanced_table(items)

        # Format the summary with key metrics
        formatted_summary = f"""
## Inventory Status Overview

Total Items Tracked: {len(items)}
Items Requiring Attention: {sum(1 for item in items if item['status'] in ['critical', 'warning'])}
Average Stock Level: {trends['average_stock_ratio']:.1%} of reorder points

### Quick Stats:
- Critical Items: {trends['critical_percentage']:.1f}%
- Warning Items: {trends['warning_percentage']:.1f}%
- Healthy Items: {100 - trends['critical_percentage'] - trends['warning_percentage']:.1f}%
"""

        return FormattedResponse(
            message_type="inventory",
            summary=formatted_summary,
            structured_data={"items": items},
            tables=[table],
            charts=charts,
            insights=insights,
            trends=trends
        )

    def _create_enhanced_charts(self, items: List[Dict], analysis: Optional[Dict]) -> List[Dict]:
        """
        Create enhanced charts based on data and AI suggestions
        """
        charts = []
        
        # Stock levels chart
        charts.append({
            "type": "bar",
            "title": "Current Stock vs Reorder Point",
            "labels": [item["name"] for item in items],
            "datasets": [
                {
                    "label": "Current Stock",
                    "data": [item["current_stock"] for item in items],
                    "backgroundColor": "rgba(54, 162, 235, 0.6)"
                },
                {
                    "label": "Reorder Point",
                    "data": [item["reorder_point"] for item in items],
                    "backgroundColor": "rgba(255, 99, 132, 0.6)"
                }
            ]
        })

        # Stock status distribution
        status_counts = {"critical": 0, "warning": 0, "good": 0}
        for item in items:
            status_counts[item["status"]] += 1

        charts.append({
            "type": "doughnut",
            "title": "Inventory Status Distribution",
            "labels": ["Critical", "Warning", "Good"],
            "datasets": [{
                "data": list(status_counts.values()),
                "backgroundColor": [
                    "rgba(255, 99, 132, 0.8)",
                    "rgba(255, 206, 86, 0.8)",
                    "rgba(75, 192, 192, 0.8)"
                ]
            }]
        })

        # Add AI-suggested visualizations
        if analysis and 'suggested_visualizations' in analysis:
            for viz in analysis['suggested_visualizations']:
                if viz['type'] not in ['bar', 'doughnut']:  # Avoid duplicates
                    charts.append({
                        "type": viz['type'],
                        "title": viz['title'],
                        "description": viz['description'],
                        "data": self._generate_chart_data(items, viz['type'])
                    })

        return charts

    def _calculate_trends(self, items: List[Dict]) -> Dict:
        """
        Calculate trends and patterns in the data
        """
        total_items = len(items)
        critical_items = sum(1 for item in items if item["status"] == "critical")
        warning_items = sum(1 for item in items if item["status"] == "warning")
        
        avg_stock_ratio = np.mean([
            item["current_stock"] / item["reorder_point"] 
            for item in items
        ])

        return {
            "total_items": total_items,
            "critical_percentage": (critical_items / total_items) * 100,
            "warning_percentage": (warning_items / total_items) * 100,
            "average_stock_ratio": avg_stock_ratio,
            "timestamp": datetime.now().isoformat()
        }

    def _create_enhanced_table(self, items: List[Dict]) -> Dict:
        """
        Create an enhanced table with additional metrics and status indicators
        """
        # Calculate additional metrics for each item
        enhanced_items = []
        for item in items:
            stock_ratio = round(item["current_stock"] / item["reorder_point"], 2)
            risk_level = self._calculate_risk_level(item)
            
            # Create status indicator
            if item["status"] == "critical":
                status_indicator = "🔴"
            elif item["status"] == "warning":
                status_indicator = "🟡"
            else:
                status_indicator = "🟢"
            
            enhanced_item = {
                "Product ID": item["product_id"],
                "Name": item["name"],
                "Current Stock": f"{item['current_stock']:,}",
                "Reorder Point": f"{item['reorder_point']:,}",
                "Stock Level": f"{stock_ratio:.0%}",
                "Status": f"{status_indicator} {item['status'].title()}",
                "Risk Level": risk_level,
                "Lead Time": item["lead_time"],
                "Location": item.get("location", "N/A"),
                "Supplier": item["supplier"]
            }
            enhanced_items.append(enhanced_item)

        return {
            "title": "Current Inventory Status",
            "description": "Detailed view of inventory items with status indicators and risk levels",
            "headers": [
                "Product ID", "Name", "Current Stock", "Reorder Point",
                "Stock Level", "Status", "Risk Level", "Lead Time",
                "Location", "Supplier"
            ],
            "rows": enhanced_items,
            "style": {
                "status_colors": {
                    "critical": "#ff4444",
                    "warning": "#ffbb33",
                    "good": "#00C851"
                }
            }
        }

    @staticmethod
    def _calculate_risk_level(item: Dict) -> str:
        """
        Calculate risk level based on multiple factors
        """
        stock_ratio = item["current_stock"] / item["reorder_point"]
        lead_time_days = int(item["lead_time"].split()[0])
        
        if stock_ratio < 0.5 and lead_time_days > 7:
            return "High Risk"
        elif stock_ratio < 0.7 or lead_time_days > 14:
            return "Medium Risk"
        else:
            return "Low Risk"

    @staticmethod
    def _determine_status(current_stock: int, reorder_point: int) -> str:
        """
        Determine the status of an inventory item based on stock levels
        """
        if current_stock <= reorder_point * 0.5:
            return "critical"
        elif current_stock <= reorder_point:
            return "warning"
        else:
            return "good"

    def _format_transport_response(self, raw_response: str) -> FormattedResponse:
        """
        Format transport-specific responses with analytics
        """
        # Get AI analysis for transport data
        analysis = self._get_ai_analysis(raw_response)
        
        if analysis:
            return FormattedResponse(
                message_type="transport",
                summary=analysis.get('summary', raw_response),
                structured_data=analysis.get('structured_data'),
                insights=analysis.get('insights', []),
                charts=analysis.get('suggested_visualizations', [])
            )
        
        return FormattedResponse(
            message_type="transport",
            summary=raw_response
        )

    def _format_general_response(self, raw_response: str) -> FormattedResponse:
        """
        Format general responses without any special formatting
        """
        # Extract response text from dictionary if needed
        if isinstance(raw_response, dict):
            response_text = raw_response.get('response', '')
            if isinstance(response_text, dict):
                response_text = response_text.get('response', '')
        else:
            response_text = str(raw_response)
        
        # For general messages, return the response as is without any special formatting
        return FormattedResponse(
            message_type="general",
            summary=response_text,
            structured_data=None,
            tables=None,
            charts=None,
            insights=None,
            trends=None
        )

    @staticmethod
    def _generate_chart_data(items: List[Dict], chart_type: str) -> Dict:
        """
        Generate data for different chart types
        """
        if chart_type == "line":
            return {
                "labels": [item["name"] for item in items],
                "datasets": [{
                    "label": "Stock Ratio Trend",
                    "data": [round(item["current_stock"] / item["reorder_point"], 2) for item in items],
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "tension": 0.1
                }]
            }
        # Add more chart types as needed
        return {} 